"""Wing / fuselage geometry morphing — the logic that actually deforms the CAD.

The core deformation lives in :func:`apply_wing_morph` and
:func:`apply_fuselage_morph`, which mutate the live TiGL ``CCPACSConfiguration``
bound to a session so downstream meshing (``generate_volume_mesh`` /
``exportWingBREPByUID``) and CFD see the new shape. These are exposed both as
standalone MCP tools (``morph_wing`` / ``morph_fuselage``) AND reused by
``set_high_level_parameters`` so that recording design intent also deforms the
geometry — closing the aero<->geometry loop without any change to the agents.

Mechanism (validated on the D150):
  1. Fetch the live ``CCPACSConfiguration`` from the session handle via
     ``CCPACSConfigurationManager.get_instance().get_configuration(handle_index)``.
  2. Scale the component transformation (wing: x=chord, y=span, z=thickness —
     span ∝ y, reference_area ∝ x·y, exact; fuselage: x=length, y·z=diameter),
     then ``WriteCPACS`` back to TiXI and **reopen a fresh TiGL handle** (the C-API
     caches ``wingGetReferenceArea`` and ``Invalidate`` does not bust it). Swap the
     fresh handle into the session. A wing's quarter-chord sweep is then set by
     shearing its positionings, and the targets are re-measured until met.
  3. Report the MEASURED geometry (wings: symmetry-aware full-planform area, so
     aviary's AREA / ASPECT_RATIO map directly; fuselage: BREP bounding box,
     avoiding the segfaulting native fuselage queries).

In stub mode (no native TiGL) the functions record intent only and report
``rebuilt=False`` so unit tests and offline callers still work.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

from tigl_mcp.cpacs import ComponentDefinition, TiglConfiguration, TixiDocument
from tigl_mcp.errors import MCPError, raise_component_not_found, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session

# Avion B252: how closely a wing morph must reach its targets, and how many scale + shear passes it may take.
_AREA_AR_TOL = 1e-4
_SWEEP_TOL_DEG = 0.01
_MORPH_PASSES = 5


class MorphWingParams(ToolParameters):
    """Parameters for morph_wing."""

    session_id: str
    wing_uid: str
    target_area_m2: float | None = None
    target_aspect_ratio: float | None = None
    target_sweep_deg: float | None = None


class MorphFuselageParams(ToolParameters):
    """Parameters for morph_fuselage."""

    session_id: str
    fuselage_uid: str
    target_length_m: float | None = None
    target_diameter_m: float | None = None


def _call_first(obj: Any, names: list[str], *args: Any) -> Any:
    """Call the first present method (handles SWIG naming differences)."""
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method(*args)
    raise AttributeError(f"none of {names} found on {type(obj).__name__}")


def _measure(
    tigl: Any, wing_uid: str, wing_index: int, sym_factor: float = 1.0
) -> dict[str, float | None]:
    """Read span / reference area / AR / sweep / MAC from a TiGL handle.

    ``sym_factor`` accounts for mirror-symmetric wings: CPACS models one half
    (e.g. ``symmetry="x-z-plane"``), so ``wingGetReferenceArea`` returns the
    half-wing area. Multiplying by 2 gives the FULL planform area — the
    convention aviary's AREA / ASPECT_RATIO design variables use. ``wingGetSpan``
    already returns the full (tip-to-tip) span, so AR = full_span**2 / full_area.
    """
    span: float = tigl.wingGetSpan(wing_uid)
    reference_area: float = tigl.wingGetReferenceArea(wing_index, 0) * sym_factor
    aspect_ratio = (span**2) / reference_area if reference_area else None
    try:
        mac_length = tigl.wingGetMAC(wing_uid)[0]
    except Exception:  # noqa: BLE001
        mac_length = None
    # Measured planform (TiGL has no sweep query): the ACHIEVED sweep and taper, not the request.
    from tigl_mcp.tools.planform import measure_planform

    planform = measure_planform(tigl, wing_index) or {}
    return {
        "span": span,
        "reference_area": reference_area,
        "aspect_ratio": aspect_ratio,
        "sweep_deg": planform.get("sweep_quarter_chord_deg"),
        "sweep_le_deg": planform.get("sweep_le_deg"),
        "taper_ratio": planform.get("taper_ratio"),
        "mac_length": mac_length,
    }


def _shear_positionings(tixi: Any, wing_uid: str, shear: float) -> None:
    """Shear the wing in its own axes: every section moves aft by ``shear`` x its spanwise position.

    Only the positionings change, so span, area, chords and taper stay as they are and a kinked planform
    keeps its kink (Avion B136: writing one sweep angle to every positioning erased the 0 deg inboard panel
    and reshaped the wing). A positioning is length L, sweep s, dihedral d, spanning
    dx = L sin s, dy = L cos s cos d, dz = L cos s sin d. TiGL's Python API has setters but no getters for
    these, so they are read and written in the CPACS document; the caller reopens TiGL from it.
    """
    base = f"{tixi.uIDGetXPath(wing_uid)}/positionings"
    for i in range(1, int(tixi.getNamedChildrenCount(base, "positioning")) + 1):
        path = f"{base}/positioning[{i}]"
        length = tixi.getDoubleElement(f"{path}/length")
        sweep = math.radians(tixi.getDoubleElement(f"{path}/sweepAngle"))
        dihedral = math.radians(tixi.getDoubleElement(f"{path}/dihedralAngle"))
        dx = length * math.sin(sweep)
        dy = length * math.cos(sweep) * math.cos(dihedral)
        dz = length * math.cos(sweep) * math.sin(dihedral)
        if abs(dy) < 1e-12:
            continue  # a section with no spanwise offset does not move under a shear
        dx += shear * dy
        tixi.updateDoubleElement(f"{path}/length", math.sqrt(dx**2 + dy**2 + dz**2), "%.12g")
        # The dihedral angle is unchanged: dy and dz keep their ratio.
        tixi.updateDoubleElement(f"{path}/sweepAngle", math.degrees(math.atan2(dx, math.hypot(dy, dz))), "%.12g")


def _fuselage_dims(tigl: Any, fuselage_uid: str) -> dict[str, float | None]:
    """Measure fuselage length/width/height from the exported BREP bounding box,
    avoiding the native fuselage geometry queries (``fuselageGetVolume`` /
    ``fuselageGetLength``) which segfault on some CPACS files (e.g. the D150)."""
    import os
    import tempfile

    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepTools import breptools_Read
        from OCC.Core.TopoDS import TopoDS_Shape
    except Exception:  # noqa: BLE001
        return {"length": None, "width": None, "height": None}

    path = tempfile.mktemp(suffix=".brep")
    try:
        tigl.exportFuselageBREPByUID(fuselage_uid, path)
        shape = TopoDS_Shape()
        breptools_Read(shape, path, BRep_Builder())
        box = Bnd_Box()
        brepbndlib_Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return {"length": xmax - xmin, "width": ymax - ymin, "height": zmax - zmin}
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass


def _reopen_handle(tixi_handle: TixiDocument, tigl_handle: TiglConfiguration, native: Any) -> Any:
    """WriteCPACS is assumed done; reopen a fresh TiGL handle from the (modified)
    TiXI doc and swap it into the session wrapper. Returns the new native handle.
    Raises on failure so callers can decide how to report it."""
    tigl_wrapper = import_module("tigl3.tigl3wrapper")
    new_native = tigl_wrapper.Tigl3()
    new_native.open(tixi_handle._tixi_handle, "")
    try:
        native.close()
    except Exception:  # noqa: BLE001
        pass
    tigl_handle._tigl_handle = new_native
    return new_native


def apply_wing_morph(
    tixi_handle: TixiDocument,
    tigl_handle: TiglConfiguration,
    component: ComponentDefinition,
    *,
    target_area_m2: float | None,
    target_aspect_ratio: float | None,
    target_sweep_deg: float | None,
) -> dict[str, object]:
    """Deform a wing to the given full-planform targets and rebuild. Shared by
    the morph_wing tool and set_high_level_parameters."""
    requested = {
        "target_area_m2": target_area_m2,
        "target_aspect_ratio": target_aspect_ratio,
        "target_sweep_deg": target_sweep_deg,
    }
    native = tigl_handle._tigl_handle
    if native is None:
        for key, value in requested.items():
            if value is not None:
                component.parameters[key.replace("target_", "").replace("_m2", "")] = value
        return {"wing_uid": component.uid, "requested": requested, "rebuilt": False,
                "note": "No native TiGL runtime — recorded intent only."}

    cfg_mod = import_module("tigl3.configuration")
    geom_mod = import_module("tigl3.geometry")
    mgr = _call_first(
        cfg_mod,
        ["CCPACSConfigurationManager_get_instance", "CCPACSConfigurationManager.get_instance"],
    )

    def wing_scaling() -> tuple[Any, Any, Any]:
        """The live configuration, its transformation and current scaling (the handle changes on reopen)."""
        cpacs_config = mgr.get_configuration(native._handle.value)
        wing = _call_first(cpacs_config, ["get_wing", "getWing"], component.uid)
        transformation = _call_first(wing, ["get_transformation", "getTransformation"])
        return cpacs_config, transformation, _call_first(transformation, ["get_scaling", "getScaling"])

    sym_factor = 2.0 if component.symmetry else 1.0
    before = _measure(native, component.uid, component.index, sym_factor)

    # Avion B252: what the caller did not ask to change is held at its measured value. A change of aspect
    # ratio scales chord and span differently, which on its own would shear the wing and lower its sweep.
    target_area = target_area_m2 if target_area_m2 is not None else float(before["reference_area"])
    target_ar = target_aspect_ratio if target_aspect_ratio is not None else float(before["aspect_ratio"] or 0.0)
    target_sweep = target_sweep_deg if target_sweep_deg is not None else before.get("sweep_deg")

    def missed(m: dict[str, float | None]) -> dict[str, bool]:
        return {
            "scale": abs(float(m["reference_area"]) / target_area - 1.0) > _AREA_AR_TOL
            or abs(float(m["aspect_ratio"]) / target_ar - 1.0) > _AREA_AR_TOL,
            "sweep": target_sweep is not None and m.get("sweep_deg") is not None
            and abs(float(m["sweep_deg"]) - float(target_sweep)) > _SWEEP_TOL_DEG,
        }

    # Scaling and shearing each move the other's measurement slightly (the wing is rotated in its incidence and
    # scaling is along its own axes), so measure after each and repeat until every target is met.
    after = before
    try:
        for _ in range(_MORPH_PASSES):
            miss = missed(after)
            if not (miss["scale"] or miss["sweep"]):
                break
            if miss["scale"]:
                span_ratio = math.sqrt(target_ar * target_area) / float(after["span"])
                area_ratio = target_area / float(after["reference_area"])
                x_ratio = area_ratio / span_ratio
                cpacs_config, transformation, cur = wing_scaling()
                _call_first(
                    transformation,
                    ["set_scaling", "setScaling"],
                    geom_mod.CTiglPoint(cur.x * x_ratio, cur.y * span_ratio, cur.z * x_ratio),
                )
                try:
                    _call_first(cpacs_config, ["invalidate", "Invalidate"])
                except AttributeError:
                    pass
                _call_first(cpacs_config, ["write_cpacs", "WriteCPACS"], _call_first(cpacs_config, ["get_uid", "GetUID"]))
                native = _reopen_handle(tixi_handle, tigl_handle, native)
                after = _measure(native, component.uid, component.index, sym_factor)
            if missed(after)["sweep"]:
                # World tan(sweep) = (scale x / scale y) x local tan(sweep), so a local shear k adds k x sx / sy.
                _, _, cur = wing_scaling()
                change = math.tan(math.radians(float(target_sweep))) - math.tan(math.radians(float(after["sweep_deg"])))
                _shear_positionings(tixi_handle._tixi_handle, component.uid, change * cur.y / cur.x)
                native = _reopen_handle(tixi_handle, tigl_handle, native)
                after = _measure(native, component.uid, component.index, sym_factor)
    except Exception as exc:  # noqa: BLE001
        after = _measure(native, component.uid, component.index, sym_factor)
        return {"wing_uid": component.uid, "requested": requested, "before": before,
                "after": after, "rebuilt": False,
                "note": f"morph applied but rebuild failed ({exc})."}

    component.parameters["span"] = after["span"]
    component.parameters["area"] = after["reference_area"]
    if after.get("aspect_ratio") is not None:
        component.parameters["aspect_ratio"] = after["aspect_ratio"]
    if after.get("sweep_deg") is not None:
        component.parameters["sweep"] = after["sweep_deg"]
    final_miss = missed(after)
    return {"wing_uid": component.uid, "requested": requested, "before": before,
            "after": after, "rebuilt": True,
            "targets": {"area": target_area, "aspect_ratio": target_ar, "sweep_quarter_chord_deg": target_sweep},
            "targets_met": not (final_miss["scale"] or final_miss["sweep"])}


def apply_fuselage_morph(
    tixi_handle: TixiDocument,
    tigl_handle: TiglConfiguration,
    component: ComponentDefinition,
    *,
    target_length_m: float | None,
    target_diameter_m: float | None,
) -> dict[str, object]:
    """Deform a fuselage to a target length/diameter and rebuild. Shared by the
    morph_fuselage tool and set_high_level_parameters."""
    requested = {"target_length_m": target_length_m, "target_diameter_m": target_diameter_m}
    native = tigl_handle._tigl_handle
    if native is None:
        for key, value in requested.items():
            if value is not None:
                component.parameters[key.replace("target_", "").replace("_m", "")] = value
        return {"fuselage_uid": component.uid, "requested": requested, "rebuilt": False,
                "note": "No native TiGL runtime — recorded intent only."}

    cfg_mod = import_module("tigl3.configuration")
    geom_mod = import_module("tigl3.geometry")
    mgr = _call_first(
        cfg_mod,
        ["CCPACSConfigurationManager_get_instance", "CCPACSConfigurationManager.get_instance"],
    )
    cpacs_config = mgr.get_configuration(native._handle.value)
    fuselage = _call_first(cpacs_config, ["get_fuselage", "getFuselage"], component.uid)

    before = _fuselage_dims(native, component.uid)
    cur_len = before["length"]
    cur_dia = (
        max(before["width"], before["height"])
        if before["width"] is not None and before["height"] is not None
        else None
    )
    if cur_len is None or cur_dia is None:
        raise_mcp_error(
            "MorphError",
            "Could not measure current fuselage dimensions (OpenCASCADE unavailable).",
        )

    x_ratio = target_length_m / cur_len if target_length_m else 1.0
    d_ratio = target_diameter_m / cur_dia if target_diameter_m else 1.0

    transformation = _call_first(fuselage, ["get_transformation", "getTransformation"])
    cur = _call_first(transformation, ["get_scaling", "getScaling"])
    _call_first(
        transformation,
        ["set_scaling", "setScaling"],
        geom_mod.CTiglPoint(cur.x * x_ratio, cur.y * d_ratio, cur.z * d_ratio),
    )
    try:
        _call_first(cpacs_config, ["invalidate", "Invalidate"])
    except AttributeError:
        pass
    uid = _call_first(cpacs_config, ["get_uid", "GetUID"])
    _call_first(cpacs_config, ["write_cpacs", "WriteCPACS"], uid)

    try:
        native = _reopen_handle(tixi_handle, tigl_handle, native)
    except Exception as exc:  # noqa: BLE001
        after = _fuselage_dims(native, component.uid)
        return {"fuselage_uid": component.uid, "requested": requested, "before": before,
                "after": after, "rebuilt": False, "note": f"reopen failed ({exc})."}

    after = _fuselage_dims(native, component.uid)
    if after.get("length") is not None:
        component.parameters["length"] = after["length"]
    if after.get("width") is not None and after.get("height") is not None:
        component.parameters["max_diameter"] = max(after["width"], after["height"])
    return {"fuselage_uid": component.uid, "requested": requested, "before": before,
            "after": after, "rebuilt": True}


def morph_wing_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the morph_wing tool — deform a wing and rebuild its geometry."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = MorphWingParams.model_validate(raw_params)
            tixi_handle, tigl_handle, config = require_session(session_manager, params.session_id)
            component = config.find_component(params.wing_uid)
            if component is None:
                raise_component_not_found(config, params.wing_uid, "Wing")
            if all(v is None for v in (params.target_area_m2, params.target_aspect_ratio, params.target_sweep_deg)):
                raise_mcp_error(
                    "MorphError",
                    "Provide at least one of target_area_m2, target_aspect_ratio, target_sweep_deg.",
                )
            return apply_wing_morph(
                tixi_handle, tigl_handle, component,
                target_area_m2=params.target_area_m2,
                target_aspect_ratio=params.target_aspect_ratio,
                target_sweep_deg=params.target_sweep_deg,
            )
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("MorphError", "Failed to morph wing geometry", str(exc))

    return ToolDefinition(
        name="morph_wing",
        description=(
            "Deform a wing to hit a target reference area, aspect ratio, and/or "
            "quarter-chord sweep, rebuilding the TiGL geometry so downstream meshing "
            "and CFD reflect the change. A target you leave out is held at its current "
            "value. target_area_m2 / target_aspect_ratio use the full-planform "
            "convention (matches aviary's AREA / ASPECT_RATIO). Returns the MEASURED "
            "geometry read back from a fresh TiGL handle, and targets_met."
        ),
        parameters_model=MorphWingParams,
        handler=handler,
        output_schema={},
    )


def morph_fuselage_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the morph_fuselage tool — deform a fuselage and rebuild geometry."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = MorphFuselageParams.model_validate(raw_params)
            tixi_handle, tigl_handle, config = require_session(session_manager, params.session_id)
            component = config.find_component(params.fuselage_uid)
            if component is None:
                raise_component_not_found(config, params.fuselage_uid, "Fuselage")
            if params.target_length_m is None and params.target_diameter_m is None:
                raise_mcp_error(
                    "MorphError", "Provide at least one of target_length_m, target_diameter_m."
                )
            return apply_fuselage_morph(
                tixi_handle, tigl_handle, component,
                target_length_m=params.target_length_m,
                target_diameter_m=params.target_diameter_m,
            )
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("MorphError", "Failed to morph fuselage geometry", str(exc))

    return ToolDefinition(
        name="morph_fuselage",
        description=(
            "Deform a fuselage to hit a target length and/or diameter, rebuilding "
            "the TiGL geometry so downstream meshing and CFD reflect the change. "
            "Dimensions are measured from the exported BREP bounding box (the "
            "native fuselage geometry queries segfault on some CPACS files)."
        ),
        parameters_model=MorphFuselageParams,
        handler=handler,
        output_schema={},
    )
