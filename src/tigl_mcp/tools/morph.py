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
     set positioning sweep, then ``WriteCPACS`` back to TiXI and **reopen a fresh
     TiGL handle** (the C-API caches ``wingGetReferenceArea`` and ``Invalidate``
     does not bust it). Swap the fresh handle into the session.
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
    try:
        sweep_deg: float | None = tigl.wingGetSweep(wing_uid)
    except Exception:  # noqa: BLE001
        sweep_deg = None
    return {
        "span": span,
        "reference_area": reference_area,
        "aspect_ratio": aspect_ratio,
        "sweep_deg": sweep_deg,
        "mac_length": mac_length,
    }


def _apply_sweep(wing: Any, sweep_deg: float) -> None:
    """Set a uniform sweep angle on every positioning (best effort)."""
    positionings = _call_first(wing, ["get_positionings", "getPositionings"])
    count = _call_first(positionings, ["get_positioning_count", "getPositioningCount"])
    for i in range(1, int(count) + 1):
        pos = _call_first(positionings, ["get_positioning", "getPositioning"], i)
        try:
            _call_first(pos, ["set_sweep_angle", "SetSweepAngle"], sweep_deg)
        except AttributeError:
            pass


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
    cpacs_config = mgr.get_configuration(native._handle.value)
    wing = _call_first(cpacs_config, ["get_wing", "getWing"], component.uid)

    sym_factor = 2.0 if component.symmetry else 1.0
    before = _measure(native, component.uid, component.index, sym_factor)
    cur_span = float(before["span"])
    cur_area = float(before["reference_area"])
    cur_ar = (cur_span**2) / cur_area if cur_area else 0.0

    target_area = target_area_m2 if target_area_m2 is not None else cur_area
    target_ar = target_aspect_ratio if target_aspect_ratio is not None else cur_ar

    target_span = math.sqrt(target_ar * target_area)
    span_ratio = target_span / cur_span if cur_span else 1.0
    area_ratio = target_area / cur_area if cur_area else 1.0
    y_ratio = span_ratio
    x_ratio = area_ratio / span_ratio if span_ratio else 1.0

    transformation = _call_first(wing, ["get_transformation", "getTransformation"])
    cur = _call_first(transformation, ["get_scaling", "getScaling"])
    _call_first(
        transformation,
        ["set_scaling", "setScaling"],
        geom_mod.CTiglPoint(cur.x * x_ratio, cur.y * y_ratio, cur.z * x_ratio),
    )
    if target_sweep_deg is not None:
        _apply_sweep(wing, target_sweep_deg)

    try:
        _call_first(cpacs_config, ["invalidate", "Invalidate"])
    except AttributeError:
        pass
    uid = _call_first(cpacs_config, ["get_uid", "GetUID"])
    _call_first(cpacs_config, ["write_cpacs", "WriteCPACS"], uid)

    try:
        native = _reopen_handle(tixi_handle, tigl_handle, native)
    except Exception as exc:  # noqa: BLE001
        after = _measure(native, component.uid, component.index, sym_factor)
        return {"wing_uid": component.uid, "requested": requested, "before": before,
                "after": after, "rebuilt": False,
                "note": f"morph applied but reopen failed ({exc})."}

    after = _measure(native, component.uid, component.index, sym_factor)
    component.parameters["span"] = after["span"]
    component.parameters["area"] = after["reference_area"]
    if after.get("aspect_ratio") is not None:
        component.parameters["aspect_ratio"] = after["aspect_ratio"]
    if after.get("sweep_deg") is not None:
        component.parameters["sweep"] = after["sweep_deg"]
    return {"wing_uid": component.uid, "requested": requested, "before": before,
            "after": after, "rebuilt": True}


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
            "sweep, rebuilding the TiGL geometry so downstream meshing and CFD "
            "reflect the change. target_area_m2 / target_aspect_ratio use the "
            "full-planform convention (matches aviary's AREA / ASPECT_RATIO). "
            "Returns the MEASURED geometry read back from a fresh TiGL handle."
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
