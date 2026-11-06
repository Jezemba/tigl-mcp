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

_REC_DTYPE = [("comp", "i4"), ("face", "i4"), ("u", "f8"), ("v", "f8"), ("mirror", "i1"), ("dist", "f8")]
_JUNCTION_DTYPE = [("node", "i4"), ("wing", "i4"), ("fuselage", "i4"), ("g", "f8"), ("xsi", "f8"), ("top", "i1")]


class DeformationError(RuntimeError):
    """The design cannot be reached by deforming the anchor mesh (reason in the message)."""


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
    inside = BRepClass3d_SolidClassifier(comp["shape"], gp_Pnt(*map(float, p)), 1e-7).State() == TopAbs_IN
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

    # Always the same procedure to full precision (no early stop on a tolerance): the anchor's own junction point is
    # computed with it too, and a node moves by the DIFFERENCE, so any procedural difference would show up as a
    # spurious displacement of the anchor onto itself.
    g0 = float(j["g"])
    f0 = f(g0)
    if f0 == 0.0:
        return _wing_point(tigl, wing["index"], g0, float(j["xsi"]), int(j["top"]), nseg), 0.0
    # Bracket outwards from g0 with geometrically growing steps, clamped to the wing's [0, nseg]: a root junction
    # sits close to g = 0, so a fixed step that skips "g < 0" can step over the crossing entirely (seen on Wing3V,
    # g0 0.0186, crossing at ~0.018).
    lo = hi = None
    d = 0.001
    while lo is None and d <= search * 1.5:
        for g in (max(0.0, g0 - d), min(float(nseg), g0 + d)):
            if g != g0 and np.sign(f(g)) != np.sign(f0):
                lo, hi = (g, g0) if g < g0 else (g0, g)
                break
        d *= 1.5
    if lo is None:
        raise DeformationError(f"no wing-fuselage intersection near the anchor junction (g {g0:.3f}, "
                               f"xsi {float(j['xsi']):.3f}) -- the design is too far from the anchor")
    flo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or hi - lo < 1e-12:
            break
        lo, hi, flo = (mid, hi, fm) if np.sign(fm) == np.sign(flo) else (lo, mid, flo)
    pt = _wing_point(tigl, wing["index"], 0.5 * (lo + hi), float(j["xsi"]), int(j["top"]), nseg)
    return pt, abs(_signed_distance(pt, fus))


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
    junction_xyz = np.array([slide_junction(handle[1], comps[int(j["wing"])], comps[int(j["fuselage"])], j)[0]
                             for j in junctions]).reshape(-1, 3)
    np.savez(out / MAP_FILE, ids=ids, rec=rec, anchor_xyz=points[ids], junctions=junctions, junction_xyz=junction_xyz,
             comp_names=np.array([c["name"] for c in comps]), face_counts=np.array([len(c["surfaces"]) for c in comps]))
    meta = {"nelem": nelem, "npoin": int(len(points)), "aircraft_nodes": int(len(ids)),
            "junction_nodes": int(len(junctions)), "map_distance_max_m": float(rec["dist"].max()),
            "components": [c["name"] for c in comps], "seconds": round(time.time() - t0, 1),
            "source_mesh": str(mesh_path), "source_cpacs": str(cpacs_path)}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))
    return meta


def surface_positions(anchor_dir: str | os.PathLike, target_tigl: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    """(global node ids, new positions, stats) of the anchor's aircraft nodes on the target geometry."""
    d = Path(anchor_dir)
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

    residual = []
    for j, at_anchor in zip(junctions, data["junction_xyz"]):
        k = int(j["node"])
        pt, res = slide_junction(target_tigl, comps[int(j["wing"])], comps[int(j["fuselage"])], j)
        shift = pt - at_anchor                      # in the unmirrored frame
        new[k] = anchor[k] + (_mirror(shift) if rec["mirror"][k] else shift)
        residual.append(res)
    moved = np.linalg.norm(new - anchor, axis=1)
    stats = {"surface_nodes": int(len(ids)), "surface_displacement_max_m": float(moved.max()),
             "surface_displacement_mean_m": float(moved.mean()), "cap_nodes": int(on_cap.sum()),
             "junction_nodes": int(len(junctions)),
             "junction_residual_max_m": float(max(residual)) if residual else 0.0}
    del anchor_handle
    return ids, new, stats


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
        raise DeformationError(f"SU2_DEF produced inverted cells (min volume {info['min_volume']}); the design "
                               f"is too far from the anchor mesh. Log: {work / 'def.log'}")
    return (work / "mesh_def.su2").read_bytes(), info


def deformed_mesh_for(target_tigl: Any, anchor_dir: str | os.PathLike) -> tuple[bytes, dict]:
    """The anchor mesh deformed to target_tigl's geometry, and the statistics for the tool reply."""
    meta = json.loads((Path(anchor_dir) / META_FILE).read_text())
    ids, xyz, stats = surface_positions(anchor_dir, target_tigl)
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
