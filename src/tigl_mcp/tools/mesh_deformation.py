"""Deform ONE anchor volume mesh to the session's current geometry instead of re-meshing it (Avion B241).

Why: re-meshing every design changes the gmsh mesh topology, and SU2's Euler drag changes with it by +-20-30 %
for geometry changes of 0.25-2.5 % (B241 ladder, 2026-11-06). Deforming one anchor mesh keeps the topology, so drag
differences follow the design (prototype: smooth and near-linear on the same ladders).

How:
1. ``prepare_anchor`` (once per study): every node of the ``aircraft`` marker of the anchor mesh is projected onto
   the component lofts of the anchor CPACS (wings with their mirrored half, fuselages); its (component, face, u, v,
   mirrored) is stored. Lofts per component, never the fused shape (boolean ops re-split faces).
2. ``surface_positions`` (per design): the same face is evaluated at the same (u, v) on the session's lofts; the node
   moves by S_new(u,v) - S_anchor(u,v), so the anchor maps onto itself exactly.
   * Planar cap faces (wing tips): a plane's (u, v) are metres from an origin that moves with the morph, so cap
     nodes take the mean displacement of their nearest curved-face nodes (same component and side).
   * Wing-body junction: nodes of a wing that sit on a fuselage in the anchor are slid along the wing's own
     spanwise coordinate (TiGL segment/eta at the node's xsi) to the new wing-fuselage intersection, so the root
     neither leaves the body nor sinks into it.
3. ``deform_mesh``: SU2_DEF (``DV_KIND= SURFACE_FILE``) with linear elasticity; refused, never silently re-meshed,
   when the face structure differs from the anchor or SU2_DEF reports negative-volume elements.

Server configuration (environment): ``TIGL_MESH_DEFORM_ANCHOR_DIR`` (directory written by ``prepare_anchor``) turns
the mode on; ``TIGL_SU2_DEF`` is the SU2_DEF executable; ``TIGL_SU2_DEF_LAUNCH`` an optional launcher prefix
(e.g. ``mpirun -np 3 --bind-to none``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ANCHOR_ENV = "TIGL_MESH_DEFORM_ANCHOR_DIR"
SU2_DEF_ENV = "TIGL_SU2_DEF"
LAUNCH_ENV = "TIGL_SU2_DEF_LAUNCH"
AIRCRAFT_MARKER = "aircraft"
BBOX_PAD_M = 0.25            # candidate faces: node inside the face bounding box grown by this
CAP_NEIGHBOURS = 4
JUNCTION_BAND_M = 0.05       # a wing node this close to a fuselage in the anchor is a junction node
JUNCTION_TOL_M = 1e-4
MAP_FILE, MESH_FILE, CPACS_FILE, META_FILE = "anchor_map.npz", "anchor.su2", "anchor.xml", "anchor.json"
# Bump when the junction procedure changes: an anchor is only exact for the procedure that prepared it.
ANCHOR_PROCEDURE = "illinois-slide-1"

_REC_DTYPE = [("comp", "i4"), ("face", "i4"), ("u", "f8"), ("v", "f8"), ("mirror", "i1"), ("dist", "f8")]
_JUNCTION_DTYPE = [("node", "i4"), ("wing", "i4"), ("fuselage", "i4"), ("g", "f8"), ("xsi", "f8"), ("top", "i1")]


class DeformationError(RuntimeError):
    """Deformation failed. The base class is a TOOL failure (missing SU2_DEF, bad anchor files ...)."""


class DesignTooFar(DeformationError):
    """The design itself is too far from the anchor for the mesh to follow it (Jessica 2026-11-07: never blocked up
    front -- how often agents reach this is measured -- but the agent must get a message it can act on).

    ``reason`` is one plain sentence about the geometry; ``measured`` the numbers behind it."""

    def __init__(self, reason: str, measured: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.measured = measured or {}


def baseline_planform(cpacs_path: str | os.PathLike, tigl: Any, cfg: Any) -> dict:
    """Each wing's baseline area, aspect ratio (as tigl-mcp measures them) and its `sweep` setting (the largest
    positioning sweep in the CPACS -- the value set_high_level_parameters' `sweep` writes)."""
    import xml.etree.ElementTree as ET

    from tigl_mcp.tools.morph import _measure

    root = ET.parse(str(cpacs_path)).getroot()
    sweeps = {}
    for wing in root.iter("wing"):
        vals = [float(p.findtext("sweepAngle") or 0.0) for p in wing.iter("positioning")]
        sweeps[wing.get("uID")] = max(vals) if vals else None
    out = {}
    for i in range(1, cfg.get_wing_count() + 1):
        w = cfg.get_wing(i)
        uid = w.get_uid()
        m = _measure(tigl, uid, i, 2.0 if int(w.get_symmetry_axis()) != 0 else 1.0)
        out[uid] = {"area_m2": round(float(m["reference_area"]), 2), "aspect_ratio": round(float(m["aspect_ratio"]), 2),
                    "sweep_setting_deg": sweeps.get(uid)}
    return out


def agent_message(exc: DeformationError, anchor_dir: str | os.PathLike | None = None) -> str:
    """The text an agent reads when generate_volume_mesh cannot deform to its design (or the tool failed)."""
    if not isinstance(exc, DesignTooFar):
        return ("MeshDeformationToolFailure: Mesh not generated because of a TOOL failure, not because of your design: " + str(exc)
                + ". Your geometry is unchanged. Calling generate_volume_mesh again will not help.")
    base = ""
    planform = _baseline_planform_of(anchor_dir)
    uid = exc.measured.get("component") if exc.measured.get("component") in planform else "Wing1"
    w = planform.get(uid)
    if w:
        base = (f" The baseline {uid} is area {w['area_m2']} m^2, aspect ratio {w['aspect_ratio']}"
                + (f", sweep setting {w['sweep_setting_deg']} deg" if w.get("sweep_setting_deg") is not None
                   else "") + ".")
    # The MCP client receives only this text (not the error type), so the text starts with the type name: agents
    # read it and study logs count refusals by it.
    return ("MeshDeformationRefused: Mesh not generated: this design is too far from the study's baseline geometry "
            "for the CFD mesh to follow it. Every design in this study is meshed by deforming one baseline mesh, and "
            f"here {exc.reason}. Your geometry in this tigl session is unchanged, but it has no mesh, so SU2 cannot "
            "run on it. To get a mesh, change the design back toward the baseline (a smaller change in wing area, "
            f"aspect ratio, sweep or tail size) and call generate_volume_mesh again.{base}")


def _baseline_planform_of(anchor_dir: str | os.PathLike | None) -> dict:
    """The anchor's baseline planform: from its metadata, else measured from the anchor CPACS (anchors prepared
    before the metadata carried it)."""
    if not anchor_dir:
        return {}
    try:
        planform = json.loads((Path(anchor_dir) / META_FILE).read_text()).get("baseline_planform")
        if planform:
            return planform
        handle = open_cpacs(Path(anchor_dir) / CPACS_FILE)
        return baseline_planform(Path(anchor_dir) / CPACS_FILE, handle[1], handle[2])
    except Exception:          # the message must still go out without the numbers
        return {}


# ---------------------------------------------------------------------------------------------------------------
# mesh file
# ---------------------------------------------------------------------------------------------------------------

def read_su2_mesh(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Points (N x 3), {marker: sorted unique node ids}, and NELEM of an ASCII SU2 mesh."""
    with open(path) as fh:
        lines = fh.readlines()
    points: np.ndarray | None = None
    markers: dict[str, np.ndarray] = {}
    nelem = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NELEM="):
            nelem = int(s.split("=")[1].split()[0])
        elif s.startswith("NPOIN="):
            n = int(s.split("=")[1].split()[0])
            points = np.loadtxt(lines[i + 1:i + 1 + n], usecols=(0, 1, 2), ndmin=2)
            i += n
        elif s.startswith("MARKER_TAG="):
            tag = s.split("=")[1].strip()
            ne = int(lines[i + 1].split("=")[1])
            ids: set[int] = set()
            for line in lines[i + 2:i + 2 + ne]:
                ids.update(int(t) for t in line.split()[1:])
            markers[tag] = np.array(sorted(ids), dtype=int)
            i += 1 + ne
        i += 1
    if points is None:
        raise DeformationError(f"{path}: no NPOIN section")
    return points, markers, nelem


# ---------------------------------------------------------------------------------------------------------------
# geometry (tigl3 + pythonocc, imported lazily so the module loads without them)
# ---------------------------------------------------------------------------------------------------------------

def open_cpacs(path: str | os.PathLike) -> tuple[Any, Any, Any]:
    """(tixi, tigl, CCPACSConfiguration) for a CPACS file; keep the tuple alive while the configuration is used."""
    from tigl3 import configuration, tigl3wrapper
    from tixi3 import tixi3wrapper

    tixi = tixi3wrapper.Tixi3()
    tixi.open(str(path))
    tigl = tigl3wrapper.Tigl3()
    tigl.open(tixi, "")
    cfg = configuration.CCPACSConfigurationManager_get_instance().get_configuration(tigl._handle.value)
    return tixi, tigl, cfg


def configuration_of(tigl: Any) -> Any:
    """The CCPACSConfiguration behind a tigl3wrapper.Tigl3 handle."""
    from tigl3 import configuration

    return configuration.CCPACSConfigurationManager_get_instance().get_configuration(tigl._handle.value)


def components(cfg: Any) -> list[dict]:
    """Each wing and fuselage: name, kind, 1-based index, symmetric flag, and per face: surface, bbox, planar."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    items = [("wing", i, cfg.get_wing(i)) for i in range(1, cfg.get_wing_count() + 1)]
    items += [("fuselage", i, cfg.get_fuselage(i)) for i in range(1, cfg.get_fuselage_count() + 1)]
    out = []
    for kind, index, comp in items:
        shape = comp.get_loft().shape()
        surfs, boxes, planar = [], [], []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            surf = BRep_Tool.Surface(face)
            box = Bnd_Box()
            brepbndlib.Add(face, box)
            surfs.append(surf)
            boxes.append(np.array(box.Get()))
            planar.append(GeomAdaptor_Surface(surf).GetType() == GeomAbs_Plane)
            exp.Next()
        out.append({"name": f"{kind}:{comp.get_uid()}", "kind": kind, "index": index,
                    "symmetric": int(comp.get_symmetry_axis()) != 0, "surfaces": surfs, "boxes": boxes,
                    "planar": planar, "shape": shape})
    return out


def _project(p: np.ndarray, surf: Any) -> tuple[float, float, float] | None:
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt

    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)
    if proj.NbPoints() == 0:
        return None
    u, v = proj.LowerDistanceParameters()
    return proj.LowerDistance(), u, v


def _mirror(p: np.ndarray) -> np.ndarray:
    return p * np.array([1.0, -1.0, 1.0])


def map_nodes(points: np.ndarray, comps: list[dict]) -> np.ndarray:
    """(component, face, u, v, mirrored, distance) of each point on the nearest component loft."""
    rec = np.zeros(len(points), dtype=_REC_DTYPE)
    for k, p in enumerate(points):
        best = (np.inf, -1, -1, 0.0, 0.0, 0)
        for ci, comp in enumerate(comps):
            queries = [(p, 0)] + ([(_mirror(p), 1)] if comp["symmetric"] else [])
            for q, mirrored in queries:
                for fi, (surf, b) in enumerate(zip(comp["surfaces"], comp["boxes"])):
                    if np.any(q < b[:3] - BBOX_PAD_M) or np.any(q > b[3:] + BBOX_PAD_M):
                        continue
                    hit = _project(q, surf)
                    if hit is not None and hit[0] < best[0]:
                        best = (hit[0], ci, fi, hit[1], hit[2], mirrored)
        rec[k] = (best[1], best[2], best[3], best[4], best[5], best[0])
    return rec


def evaluate(rec: np.ndarray, comps: list[dict]) -> np.ndarray:
    out = np.zeros((len(rec), 3))
    for k, r in enumerate(rec):
        pt = comps[int(r["comp"])]["surfaces"][int(r["face"])].Value(float(r["u"]), float(r["v"]))
        out[k] = (pt.X(), -pt.Y() if r["mirror"] else pt.Y(), pt.Z())
    return out


def _signed_distance(p: np.ndarray, comp: dict) -> float:
    """Distance of p from a fuselage loft, negative inside the body."""
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopAbs import TopAbs_IN

    near = [s for s, b in zip(comp["surfaces"], comp["boxes"])
            if not (np.any(p < b[:3] - 1.0) or np.any(p > b[3:] + 1.0))] or comp["surfaces"]
    best = min((h[0] for h in (_project(p, s) for s in near) if h is not None), default=np.inf)
    # Loading the solid into a classifier costs ~41 ms; it was rebuilt on every call (91 % of a design's 13 min,
    # profile 2026-11-07). Load once per component and classify each point with Perform (same algorithm).
    classifier = comp.get("_classifier")
    if classifier is None:
        classifier = comp["_classifier"] = BRepClass3d_SolidClassifier(comp["shape"])
    classifier.Perform(gp_Pnt(*map(float, p)), 1e-7)
    inside = classifier.State() == TopAbs_IN
    return -best if inside else best


def _wing_point(tigl: Any, wing: int, g: float, xsi: float, top: int, nseg: int) -> np.ndarray:
    """Point on a wing at global spanwise coordinate g (segment s covers g in [s-1, s]) and chordwise xsi."""
    g = min(max(g, 0.0), float(nseg))
    seg = min(int(g) + 1, nseg)
    eta = g - (seg - 1)
    fn = tigl.wingGetUpperPoint if top else tigl.wingGetLowerPoint
    return np.array(fn(wing, seg, eta, min(max(xsi, 0.0), 1.0)))


def find_junctions(points: np.ndarray, rec: np.ndarray, comps: list[dict], tigl: Any) -> np.ndarray:
    """Wing nodes lying on a fuselage in the anchor, with their TiGL (global eta, xsi, upper/lower)."""
    out = []
    fuselages = [i for i, c in enumerate(comps) if c["kind"] == "fuselage"]
    for k, r in enumerate(rec):
        comp = comps[int(r["comp"])]
        if comp["kind"] != "wing":
            continue
        p = _mirror(points[k]) if r["mirror"] else points[k]
        for fi in fuselages:
            if abs(_signed_distance(p, comps[fi])) < JUNCTION_BAND_M:
                seg, eta, xsi, top = tigl.wingGetSegmentEtaXsi(comp["index"], *map(float, p))
                out.append((k, int(r["comp"]), fi, (seg - 1) + eta, xsi, int(top)))
                break
    return np.array(out, dtype=_JUNCTION_DTYPE)


def slide_junction(tigl: Any, wing: dict, fus: dict, j: np.void, search: float = 1.5) -> tuple[np.ndarray, float]:
    """The point on the (morphed) wing's spanwise line through (xsi, top) that lies on the fuselage surface.

    Bracket the sign change of the signed fuselage distance around the anchor's g, then bisect.
    Returns (point in the unmirrored frame, residual distance)."""
    nseg = tigl.wingGetSegmentCount(wing["index"])

    def f(g: float) -> float:
        return _signed_distance(_wing_point(tigl, wing["index"], g, float(j["xsi"]), int(j["top"]), nseg), fus)

    # Always the same deterministic procedure: the anchor's own junction point is computed with it too, and a node
    # moves by the DIFFERENCE, so any procedural difference would show up as a spurious displacement of the anchor
    # onto itself (an anchor must be prepared with the same version of this function that deforms from it).
    g0 = float(j["g"])
    f0 = f(g0)
    if f0 == 0.0:
        slide_junction.last_g = g0
        return _wing_point(tigl, wing["index"], g0, float(j["xsi"]), int(j["top"]), nseg), 0.0
    # Bracket outwards from g0 with geometrically growing steps, clamped to the wing's [0, nseg]: a root junction
    # sits close to g = 0, so a fixed step that skips "g < 0" can step over the crossing entirely (seen on Wing3V,
    # g0 0.0186, crossing at ~0.018).
    lo = hi = None
    d = 0.001
    while lo is None and d <= search * 1.5:
        for g in (max(0.0, g0 - d), min(float(nseg), g0 + d)):
            if g == g0:
                continue
            fg = f(g)
            if np.sign(fg) != np.sign(f0):
                (lo, flo), (hi, fhi) = ((g, fg), (g0, f0)) if g < g0 else ((g0, f0), (g, fg))
                break
        d *= 1.5
    if lo is None:
        name = wing["name"].split(":")[-1]
        raise DesignTooFar(f"the root of {name} no longer meets the fuselage where the baseline mesh joins them",
                           {"component": name, "junction_g": round(g0, 4), "junction_xsi": round(float(j["xsi"]), 4)})
    g_hit = _illinois(f, lo, flo, hi, fhi)
    pt = _wing_point(tigl, wing["index"], g_hit, float(j["xsi"]), int(j["top"]), nseg)
    slide_junction.last_g = g_hit                  # the intersection's spanwise coordinate, for the root stretch
    return pt, abs(_signed_distance(pt, fus))


SLIDE_TOL_M = 1e-7          # = the solid classifier's tolerance: the distance cannot be resolved more finely
SLIDE_TOL_G = 1e-12


def _illinois(f: Any, lo: float, flo: float, hi: float, fhi: float, maxiter: int = 100) -> float:
    """Root of f in the sign-change bracket [lo, hi] by Illinois regula falsi (always bracketed, superlinear).

    Plain bisection to 1e-12 took ~39 signed-distance evaluations per junction slide, each with a fuselage
    inside/outside classification (profile 2026-11-07: 91 % of a design's 13 min); the signed distance is smooth
    across the surface, so a secant-type step needs a handful."""
    side = 0
    mid = 0.5 * (lo + hi)
    for _ in range(maxiter):
        mid = (lo * fhi - hi * flo) / (fhi - flo)
        if not lo < mid < hi:
            mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= SLIDE_TOL_M:
            return mid
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
            if side == -1:
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = mid, fm
            if side == 1:
                flo *= 0.5
            side = 1
        if hi - lo < SLIDE_TOL_G:
            return 0.5 * (lo + hi)
    return mid


slide_junction.last_g = 0.0


_WING_PARAM_DTYPE = [("node", "i4"), ("g", "f8"), ("xsi", "f8"), ("top", "i1"), ("ok", "i1")]


def wing_params(points: np.ndarray, rec: np.ndarray, comps: list[dict], tigl: Any) -> np.ndarray:
    """TiGL (global eta g, xsi, upper/lower) of every wing node on a curved face, in the unmirrored frame.

    ok = the parameters reproduce the node within 1 mm (else the node keeps the face mapping)."""
    out = []
    for k, r in enumerate(rec):
        comp = comps[int(r["comp"])]
        if comp["kind"] != "wing" or comp["planar"][int(r["face"])]:
            continue
        p = _mirror(points[k]) if r["mirror"] else points[k]
        try:
            seg, eta, xsi, top = tigl.wingGetSegmentEtaXsi(comp["index"], *map(float, p))
        except Exception:  # noqa: BLE001 - TiGL refuses points it cannot place
            out.append((k, 0.0, 0.0, 0, 0))
            continue
        g = (seg - 1) + eta
        nseg = tigl.wingGetSegmentCount(comp["index"])
        back = _wing_point(tigl, comp["index"], g, xsi, int(top), nseg)
        out.append((k, g, xsi, int(top), int(np.linalg.norm(back - p) < 1e-3)))
    return np.array(out, dtype=_WING_PARAM_DTYPE)


# ---------------------------------------------------------------------------------------------------------------
# anchor preparation and per-design positions
# ---------------------------------------------------------------------------------------------------------------

def prepare_anchor(mesh_path: str | os.PathLike, cpacs_path: str | os.PathLike, out_dir: str | os.PathLike) -> dict:
    """Copy the anchor mesh and CPACS into out_dir and write the surface correspondence. Returns the metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copyfile(mesh_path, out / MESH_FILE)
    shutil.copyfile(cpacs_path, out / CPACS_FILE)
    points, markers, nelem = read_su2_mesh(out / MESH_FILE)
    if AIRCRAFT_MARKER not in markers:
        raise DeformationError(f"anchor mesh has no '{AIRCRAFT_MARKER}' marker ({sorted(markers)})")
    ids = markers[AIRCRAFT_MARKER]
    handle = open_cpacs(out / CPACS_FILE)
    comps = components(handle[2])
    rec = map_nodes(points[ids], comps)
    if (rec["comp"] < 0).any():
        raise DeformationError(f"{int((rec['comp'] < 0).sum())} aircraft nodes are on no component loft")
    junctions = find_junctions(points[ids], rec, comps, handle[1])
    # The anchor's own wing-fuselage intersection points: a junction node moves by (new - anchor) intersection,
    # so it keeps its offset from the junction curve and the anchor maps onto itself exactly.
    junction_xyz, junction_g = [], []
    for j in junctions:
        junction_xyz.append(slide_junction(handle[1], comps[int(j["wing"])], comps[int(j["fuselage"])], j)[0])
        junction_g.append(slide_junction.last_g)
    junction_xyz = np.array(junction_xyz).reshape(-1, 3)
    wp = wing_params(points[ids], rec, comps, handle[1])
    root_g0 = _exact_root_g(wp, rec, comps, junctions, np.array(junction_g), handle[1])
    np.savez(out / MAP_FILE, ids=ids, rec=rec, anchor_xyz=points[ids], junctions=junctions, junction_xyz=junction_xyz,
             junction_g=np.array(junction_g), wing_params=wp, root_g0=root_g0,
             tris=aircraft_triangles(out / MESH_FILE, ids),
             comp_names=np.array([c["name"] for c in comps]), face_counts=np.array([len(c["surfaces"]) for c in comps]))
    meta = {"procedure": ANCHOR_PROCEDURE, "nelem": nelem, "npoin": int(len(points)), "aircraft_nodes": int(len(ids)),
            "junction_nodes": int(len(junctions)), "map_distance_max_m": float(rec["dist"].max()),
            "components": [c["name"] for c in comps], "seconds": round(time.time() - t0, 1),
            "source_mesh": str(mesh_path), "source_cpacs": str(cpacs_path),
            "baseline_planform": baseline_planform(out / CPACS_FILE, handle[1], handle[2])}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))
    return meta


def surface_positions(anchor_dir: str | os.PathLike, target_tigl: Any,
                      strict: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """(global node ids, new positions, stats) of the anchor's aircraft nodes on the target geometry."""
    d = Path(anchor_dir)
    procedure = json.loads((d / META_FILE).read_text()).get("procedure")
    if procedure != ANCHOR_PROCEDURE:
        raise DeformationError(f"anchor {d} was prepared with junction procedure {procedure!r}, this server uses "
                               f"{ANCHOR_PROCEDURE!r}; re-run `prepare` for this anchor")
    data = np.load(d / MAP_FILE)
    rec, ids, anchor, junctions = data["rec"], data["ids"], data["anchor_xyz"], data["junctions"]
    anchor_handle = open_cpacs(d / CPACS_FILE)
    comps0 = components(anchor_handle[2])
    comps = components(configuration_of(target_tigl))
    if [c["name"] for c in comps] != list(data["comp_names"]) or \
            [len(c["surfaces"]) for c in comps] != list(data["face_counts"]):
        raise DeformationError("the geometry's components/faces differ from the anchor's "
                               f"({[(c['name'], len(c['surfaces'])) for c in comps]} vs "
                               f"{list(zip(data['comp_names'], data['face_counts']))}); deformation cannot map it")
    disp = evaluate(rec, comps) - evaluate(rec, comps0)

    on_cap = np.array([comps0[int(r["comp"])]["planar"][int(r["face"])] for r in rec])
    for i in np.where(on_cap)[0]:
        same = np.where(~on_cap & (rec["comp"] == rec["comp"][i]) & (rec["mirror"] == rec["mirror"][i]))[0]
        if len(same):
            near = same[np.argsort(np.linalg.norm(anchor[same] - anchor[i], axis=1))[:CAP_NEIGHBOURS]]
            disp[i] = disp[near].mean(axis=0)
    new = anchor + disp

    # The wing-body junction on the target: each junction node's intersection with the fuselage, by spanwise g.
    residual = []
    g_target = np.zeros(len(junctions))
    for n, j in enumerate(junctions):
        _, res = slide_junction(target_tigl, comps[int(j["wing"])], comps[int(j["fuselage"])], j)
        g_target[n] = slide_junction.last_g
        residual.append(res)
    root = _root_stretch(rec, anchor, new, data, junctions, g_target, anchor_handle[1], target_tigl, comps)
    # Cap nodes follow their (now final) curved neighbours.
    for i in np.where(on_cap)[0]:
        same = np.where(~on_cap & (rec["comp"] == rec["comp"][i]) & (rec["mirror"] == rec["mirror"][i]))[0]
        if len(same):
            near = same[np.argsort(np.linalg.norm(anchor[same] - anchor[i], axis=1))[:CAP_NEIGHBOURS]]
            new[i] = anchor[i] + (new[near] - anchor[near]).mean(axis=0)
    fus_moved = _fuselage_follow(rec, anchor, new, junctions, comps)

    tris = data["tris"] if "tris" in data.files else aircraft_triangles(d / MESH_FILE, ids)
    flipped = flipped_triangles(tris, anchor, new)
    moved = np.linalg.norm(new - anchor, axis=1)
    jmove = moved[junctions["node"].astype(int)] if len(junctions) else np.zeros(1)
    stats = {"surface_nodes": int(len(ids)), "surface_displacement_max_m": float(moved.max()),
             "surface_displacement_mean_m": float(moved.mean()), "cap_nodes": int(on_cap.sum()),
             "junction_nodes": int(len(junctions)),
             "junction_residual_max_m": float(max(residual)) if residual else 0.0,
             "junction_move_max_m": float(jmove.max()), "root_zone_nodes": root["zone"],
             "wing_param_fallback_nodes": root["fallback"], "fuselage_nodes_following_junction": fus_moved,
             "surface_triangles_flipped": int(len(flipped))}
    del anchor_handle
    stats["_flipped"] = flipped.tolist()       # for diagnosis; removed from the reply below
    if strict and len(flipped):
        # Name the component whose junction folds (nearest junction node to the folded triangles): a tail change
        # folds at the TAIL root, and "the wing root" would send the agent to the wrong component.
        name, move = _folding_component(tris[np.asarray(flipped, dtype=int)], anchor, new, junctions, comps)
        raise DesignTooFar(f"the root of {name} would move {move:.2f} m along the fuselage and the surface would "
                           f"fold where {name} joins the fuselage ({len(flipped)} of {len(tris)} surface triangles "
                           "turn over)",
                           {"component": name, "surface_triangles_flipped": int(len(flipped)),
                            "junction_move_max_m": round(move, 3),
                            "surface_displacement_max_m": round(stats["surface_displacement_max_m"], 3)})
    return ids, new, stats


def _folding_component(tris: np.ndarray, anchor: np.ndarray, new: np.ndarray, junctions: np.ndarray,
                       comps: list[dict]) -> tuple[str, float]:
    """(component UID, its largest junction move in m) for the junction closest to the folded triangles."""
    if not len(junctions):
        return "the aircraft", 0.0
    jnodes = junctions["node"].astype(int)
    centres = anchor[tris].mean(axis=1)
    nearest = np.argmin(np.linalg.norm(centres[:, None, :] - anchor[jnodes][None, :, :], axis=2), axis=1)
    wing = int(np.bincount(junctions["wing"][nearest].astype(int)).argmax())
    sel = jnodes[junctions["wing"] == wing]
    return comps[wing]["name"].split(":")[-1], float(np.linalg.norm(new[sel] - anchor[sel], axis=1).max())


ROOT_ZONE_MIN_G = 0.3          # the stretched strip next to the junction is at least this wide in g
ROOT_ZONE_FACTOR = 3.0         # ... and at least this many times the junction's own spanwise move
ROOT_EXACT_BAND_G = 0.1        # wing nodes this close (in g) to the junction use the exact intersection at their xsi


def _exact_root_g(wp: np.ndarray, rec: np.ndarray, comps: list[dict], junctions: np.ndarray, jg0: np.ndarray,
                  tigl: Any) -> np.ndarray:
    """Anchor: the exact junction g at each near-root wing node's own (xsi, top); NaN for the others."""
    out = np.full(len(wp), np.nan)
    for i, r in enumerate(wp):
        k = int(r["node"])
        ci = int(rec["comp"][k])
        jsel = np.where((junctions["wing"] == ci) & (rec["mirror"][junctions["node"]] == 0))[0]
        if not r["ok"] or not len(jsel):
            continue
        s = jsel[junctions["top"][jsel] == int(r["top"])] if (junctions["top"][jsel] == int(r["top"])).any() else jsel
        order = np.argsort(junctions["xsi"][s])
        xs = junctions["xsi"][s][order]
        gj = float(np.interp(float(r["xsi"]), xs, jg0[s][order]))
        # Exact also where the curve would EXTRAPOLATE (xsi outside its junction nodes' range): AR 7, 2026-11-07 --
        # a lower-surface node at xsi 0.02, lower junction curve starting further aft, folded 2 triangles at the LE.
        outside = float(r["xsi"]) < xs[0] or float(r["xsi"]) > xs[-1]
        if float(r["g"]) - gj < ROOT_EXACT_BAND_G or (outside and float(r["g"]) - gj < ROOT_ZONE_MIN_G):
            fus = comps[int(junctions["fuselage"][jsel[0]])]
            try:
                slide_junction(tigl, comps[ci], fus, {"g": gj, "xsi": float(r["xsi"]), "top": int(r["top"])})
            except DeformationError:
                continue                        # no intersection on this line: keep the interpolated curve
            out[i] = slide_junction.last_g
    return out


def _root_stretch(rec: np.ndarray, anchor: np.ndarray, new: np.ndarray, data: Any, junctions: np.ndarray,
                  g_target: np.ndarray, tigl0: Any, tigl1: Any, comps: list[dict]) -> dict:
    """Wing nodes move in TiGL's own wing parameters; next to the root the spanwise coordinate is STRETCHED.

    For a node at (g, xsi): gJ0 / gJ1 = the junction's g at that xsi on the anchor / target; outside a strip of
    width W from gJ0 the node keeps its g, inside it g' = gJ1 + (g - gJ0)(gout - gJ1)/(gout - gJ0), gout = gJ0 + W.
    This is monotone in g, so nodes keep their spanwise order (no fold) and the junction row lands on the body.
    Moves = P1(g', xsi) - P0(g, xsi), so the anchor maps onto itself exactly. In place."""
    if "wing_params" not in data.files:
        raise DeformationError("anchor prepared without wing parameters; re-run prepare_anchor")
    wp = data["wing_params"]
    jg0 = data["junction_g"]
    exact = ({int(n): float(g) for n, g in zip(wp["node"], data["root_g0"]) if np.isfinite(g)}
             if "root_g0" in data.files else {})
    zone = fallback = 0
    for ci, comp in enumerate(comps):
        if comp["kind"] != "wing":
            continue
        rows = wp[rec["comp"][wp["node"]] == ci]
        if not len(rows):
            continue
        nseg = tigl1.wingGetSegmentCount(comp["index"])
        jsel = np.where((junctions["wing"] == ci) & (rec["mirror"][junctions["node"]] == 0))[0]
        curves = {}
        for top in (0, 1):
            s = jsel[junctions["top"][jsel] == top]
            if len(s) == 0:
                s = jsel
            order = np.argsort(junctions["xsi"][s])
            curves[top] = (junctions["xsi"][s][order], jg0[s][order], g_target[s][order])
        width = ROOT_ZONE_MIN_G
        if len(jsel):
            width = max(width, ROOT_ZONE_FACTOR * float(np.abs(g_target[jsel] - jg0[jsel]).max()))
        for r in rows:
            k = int(r["node"])
            if not r["ok"]:
                fallback += 1
                continue
            g, xsi, top = float(r["g"]), float(r["xsi"]), int(r["top"])
            g_new = g
            if len(jsel):
                xs, g0s, g1s = curves[top]
                gj0, gj1 = float(np.interp(xsi, xs, g0s)), float(np.interp(xsi, xs, g1s))
                exact0 = exact.get(k)
                if exact0 is not None:
                    # Next to the junction the interpolated curve is not good enough at the root corners (LE/TE),
                    # where the intersection bends: use the exact intersection at this node's own xsi, both sides.
                    fus = comps[int(junctions["fuselage"][jsel[0]])]
                    probe = {"g": gj1, "xsi": xsi, "top": top}
                    slide_junction(tigl1, comp, fus, probe)
                    gj0, gj1 = exact0, slide_junction.last_g
                gout = min(gj0 + width, float(nseg))
                if g < gout and gout > gj0 and gout > gj1:
                    g_new = gj1 + max(g - gj0, 0.0) * (gout - gj1) / (gout - gj0)
                    zone += 1
            shift = _wing_point(tigl1, comp["index"], g_new, xsi, top, nseg) - \
                _wing_point(tigl0, comp["index"], g, xsi, top, nseg)
            new[k] = anchor[k] + (_mirror(shift) if rec["mirror"][k] else shift)
    return {"zone": zone, "fallback": fallback}


def _wendland(r: np.ndarray) -> np.ndarray:
    """Wendland C2 kernel on r in [0, 1] (compact support, positive definite in 3-D)."""
    r = np.clip(r, 0.0, 1.0)
    return (1.0 - r) ** 4 * (4.0 * r + 1.0)


def _fuselage_follow(rec: np.ndarray, anchor: np.ndarray, new: np.ndarray, junctions: np.ndarray,
                     comps: list[dict]) -> int:
    """Fuselage nodes near the junction follow it: a Wendland RBF interpolates the junction nodes' moves, fades to zero
    over a support of max(1 m, 4 x the largest move) -- gradients stay small, so the fuselage does not fold -- and each
    moved node is put back on the fuselage loft. In place; returns the number of nodes moved."""
    if not len(junctions):
        return 0
    jn = junctions["node"].astype(int)
    moves = new[jn] - anchor[jn]
    big = float(np.linalg.norm(moves, axis=1).max())
    if big < 1e-12:
        return 0
    support = max(1.0, 4.0 * big)
    centres = anchor[jn]
    phi = _wendland(np.linalg.norm(centres[:, None] - centres[None], axis=2) / support)
    coef = np.linalg.solve(phi + 1e-10 * np.eye(len(jn)), moves)
    jset = set(jn.tolist())
    count = 0
    for k in range(len(rec)):
        comp = comps[int(rec["comp"][k])]
        if comp["kind"] != "fuselage" or k in jset:
            continue
        r = np.linalg.norm(centres - anchor[k], axis=1) / support
        if r.min() >= 1.0:
            continue
        add = _wendland(r) @ coef
        if np.linalg.norm(add) < 1e-9:
            continue
        new[k] = _onto_component(anchor[k] + add, comp, bool(rec["mirror"][k]))
        count += 1
    return count


def _onto_component(p: np.ndarray, comp: dict, mirrored: bool) -> np.ndarray:
    """Nearest point of a component loft (in the node's own frame)."""
    q = _mirror(p) if mirrored else p
    near = [s for s, b in zip(comp["surfaces"], comp["boxes"])
            if not (np.any(q < b[:3] - 1.0) or np.any(q > b[3:] + 1.0))] or comp["surfaces"]
    best, best_pt = np.inf, q
    for surf in near:
        hit = _project(q, surf)
        if hit is not None and hit[0] < best:
            pt = surf.Value(hit[1], hit[2])
            best, best_pt = hit[0], np.array([pt.X(), pt.Y(), pt.Z()])
    return _mirror(best_pt) if mirrored else best_pt


def aircraft_triangles(mesh_path: str | os.PathLike, ids: np.ndarray) -> np.ndarray:
    """The aircraft marker's triangles as indices into ids."""
    local = {int(g): i for i, g in enumerate(ids)}
    tris = []
    with open(mesh_path) as fh:
        lines = iter(fh)
        for line in lines:
            if line.strip() == f"MARKER_TAG= {AIRCRAFT_MARKER}":
                n = int(next(lines).split("=")[1])
                for _ in range(n):
                    t = next(lines).split()
                    tris.append([local[int(t[1])], local[int(t[2])], local[int(t[3])]])
                break
    return np.array(tris, dtype=int).reshape(-1, 3)


def flipped_triangles(tris: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Indices of surface triangles whose normal reversed (the surface folded)."""
    if not len(tris):
        return np.array([], dtype=int)

    def normals(x):
        return np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])

    return np.where((normals(before) * normals(after)).sum(1) <= 0.0)[0]


# ---------------------------------------------------------------------------------------------------------------
# SU2_DEF
# ---------------------------------------------------------------------------------------------------------------

DEF_CONFIG = """\
SOLVER= EULER
MATH_PROBLEM= DIRECT
MESH_FORMAT= SU2
MESH_FILENAME= {mesh}
MESH_OUT_FILENAME= mesh_def.su2
MARKER_EULER= ( {marker} )
MARKER_FAR= ( farfield )
DV_KIND= SURFACE_FILE
DV_MARKER= ( {marker} )
DV_PARAM= ( 1.0 )
DV_VALUE= 1.0
DV_FILENAME= surface_positions.dat
DEFORM_LINEAR_SOLVER= FGMRES
DEFORM_LINEAR_SOLVER_PREC= ILU
DEFORM_LINEAR_SOLVER_ITER= 1000
DEFORM_LINEAR_SOLVER_ERROR= 1E-10
DEFORM_NONLINEAR_ITER= {nonlinear}
DEFORM_STIFFNESS_TYPE= WALL_DISTANCE
DEFORM_CONSOLE_OUTPUT= YES
"""


def write_positions(path: str | os.PathLike, ids: np.ndarray, xyz: np.ndarray) -> None:
    """SU2 SURFACE_FILE: whitespace-separated 'GlobalIndex x y z' (SU2 reads it with operator>>)."""
    with open(path, "w") as fh:
        for gid, p in zip(ids, xyz):
            fh.write(f"{int(gid)} {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")


def parse_def_log(text: str) -> dict:
    vols = re.findall(r"Min\. volume: ([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    neg = re.findall(r"(\d+)\s+elements? with negative volume|negative volume", text, flags=re.I)
    return {"min_volume": float(vols[-1]) if vols else None, "negative_volume_reported": bool(neg)}


def deform_mesh(anchor_dir: str | os.PathLike, ids: np.ndarray, xyz: np.ndarray, *, su2_def: str | None = None,
                launch: str | None = None, nonlinear_iter: int = 1, timeout_s: int = 1800) -> tuple[bytes, dict]:
    """Run SU2_DEF on the anchor mesh with the given surface positions; returns (mesh bytes, stats)."""
    su2_def = su2_def or os.environ.get(SU2_DEF_ENV) or shutil.which("SU2_DEF")
    if not su2_def:
        raise DeformationError(f"SU2_DEF not found (set {SU2_DEF_ENV})")
    launch = launch if launch is not None else os.environ.get(LAUNCH_ENV, "")
    work = Path(tempfile.mkdtemp(prefix="tigl_deform_"))
    write_positions(work / "surface_positions.dat", ids, xyz)
    (work / "def.cfg").write_text(DEF_CONFIG.format(mesh=Path(anchor_dir).resolve() / MESH_FILE,
                                                    marker=AIRCRAFT_MARKER, nonlinear=int(nonlinear_iter)))
    t0 = time.time()
    proc = subprocess.run(shlex.split(launch) + [su2_def, "def.cfg"], cwd=work, capture_output=True, text=True,
                          timeout=timeout_s)
    log = proc.stdout + proc.stderr
    (work / "def.log").write_text(log)
    info = parse_def_log(log)
    info.update({"deform_seconds": round(time.time() - t0, 1), "workdir": str(work), "exit_code": proc.returncode})
    if proc.returncode != 0 or not (work / "mesh_def.su2").exists():
        raise DeformationError(f"SU2_DEF failed (exit {proc.returncode}); log {work / 'def.log'}: {log[-800:]}")
    if info["negative_volume_reported"] or (info["min_volume"] is not None and info["min_volume"] <= 0):
        raise DesignTooFar("the volume mesh around the aircraft would turn inside out (SU2_DEF: cells with "
                           "negative volume)", {"min_volume": info["min_volume"], "log": str(work / "def.log")})
    return (work / "mesh_def.su2").read_bytes(), info


def deformed_mesh_for(target_tigl: Any, anchor_dir: str | os.PathLike) -> tuple[bytes, dict]:
    """The anchor mesh deformed to target_tigl's geometry, and the statistics for the tool reply."""
    meta = json.loads((Path(anchor_dir) / META_FILE).read_text())
    ids, xyz, stats = surface_positions(anchor_dir, target_tigl)
    stats.pop("_flipped", None)
    mesh, info = deform_mesh(anchor_dir, ids, xyz)
    stats.update(info)
    stats.update({"mesh_mode": "deformed_from_anchor", "anchor_dir": str(anchor_dir),
                  "element_count": meta["nelem"], "node_count": meta["npoin"]})
    return mesh, stats


def anchor_dir_from_env() -> str | None:
    value = os.environ.get(ANCHOR_ENV, "").strip()
    return value or None


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m tigl_mcp.tools.mesh_deformation prepare <anchor.su2> <anchor.xml> <out_dir>``."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4 or args[0] != "prepare":
        print(main.__doc__)
        return 2
    print(json.dumps(prepare_anchor(args[1], args[2], args[3]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
