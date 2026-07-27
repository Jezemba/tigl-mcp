"""Wing geometry morphing — the tool that actually deforms the CAD.

Unlike :func:`set_high_level_parameters` (which only records design intent in an
in-memory dict), ``morph_wing`` mutates the *live* TiGL ``CCPACSConfiguration``
bound to the session, so downstream meshing (``generate_volume_mesh`` /
``exportWingBREPByUID``) and CFD see the new shape.

Mechanism (validated against the installed tigl3 build on the D150):
  1. Fetch the live ``CCPACSConfiguration`` from the session's handle via
     ``CCPACSConfigurationManager.get_instance().get_configuration(handle_index)``.
  2. Measure the current span & reference area (first C-API reads are accurate).
  3. Compute a wing-transformation scaling (x=chord, y=span, z=thickness) that
     hits the requested reference area and aspect ratio. The relationship is
     exact/linear: span ∝ y, reference_area ∝ x·y (verified). Optional sweep is
     applied via the positionings' ``set_sweep_angle``.
  4. ``WriteCPACS`` the mutated model back into the session's TiXI document and
     **reopen a fresh TiGL handle** from it. The reopen is required because the
     C-API caches ``wingGetReferenceArea`` on a handle (``Invalidate`` does not
     bust it); a fresh handle reports the morphed span AND area consistently, so
     ``get_wing_summary`` on the session afterwards is correct.
  5. Swap the session's native handle to the fresh one and read back the
     measured geometry (this is also the regression-test assertion).

Only the real TiGL runtime can deform geometry; in stub mode (no native
bindings) the tool records the requested values and reports ``rebuilt=False`` so
unit tests and offline callers still get a sane response.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

from tigl_mcp.errors import MCPError, raise_mcp_error
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
    half-wing area. Multiplying by 2 gives the FULL planform area, which is the
    convention aviary's AREA / ASPECT_RATIO design variables use — so the agent
    can pass aviary's numbers directly. ``wingGetSpan`` already returns the full
    (tip-to-tip) span, so AR = full_span**2 / full_area is the true aspect ratio.
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
    """Measure fuselage length/width/height from the exported BREP's bounding
    box. This deliberately AVOIDS the native fuselage geometry queries
    (``fuselageGetVolume`` / ``fuselageGetLength``), which segfault on some CPACS
    files (e.g. the D150) and take the whole server down uncatchably. BREP export
    + an OpenCASCADE bounding box are safe."""
    import os
    import tempfile

    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepTools import breptools_Read
        from OCC.Core.TopoDS import TopoDS_Shape
    except Exception:  # noqa: BLE001 - OCC not present (stub/offline)
        return {"length": None, "width": None, "height": None}

    path = tempfile.mktemp(suffix=".brep")
    try:
        tigl.exportFuselageBREPByUID(fuselage_uid, path)
        shape = TopoDS_Shape()
        breptools_Read(shape, path, BRep_Builder())
        box = Bnd_Box()
        brepbndlib_Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return {
            "length": xmax - xmin,
            "width": ymax - ymin,
            "height": zmax - zmin,
        }
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass


def morph_wing_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the morph_wing tool — deform a wing and rebuild its geometry."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = MorphWingParams.model_validate(raw_params)
            tixi_handle, tigl_handle, config = require_session(
                session_manager, params.session_id
            )
            component = config.find_component(params.wing_uid)
            if component is None:
                raise_mcp_error("NotFound", f"Wing '{params.wing_uid}' not found")

            requested = {
                "target_area_m2": params.target_area_m2,
                "target_aspect_ratio": params.target_aspect_ratio,
                "target_sweep_deg": params.target_sweep_deg,
            }
            if all(v is None for v in requested.values()):
                raise_mcp_error(
                    "MorphError",
                    "Provide at least one of target_area_m2, "
                    "target_aspect_ratio, target_sweep_deg.",
                )

            native = tigl_handle._tigl_handle

            # ── Stub / offline mode: no native TiGL, cannot deform CAD ──────
            if native is None:
                for key, value in requested.items():
                    if value is not None:
                        pname = key.replace("target_", "").replace("_m2", "")
                        component.parameters[pname] = value
                return {
                    "wing_uid": component.uid,
                    "requested": requested,
                    "rebuilt": False,
                    "note": "No native TiGL runtime — recorded intent only; "
                    "geometry not deformed. Run against the real tigl-env server.",
                }

            # ── Real mode: mutate the live CCPACSConfiguration ──────────────
            cfg_mod = import_module("tigl3.configuration")
            geom_mod = import_module("tigl3.geometry")
            tigl_wrapper = import_module("tigl3.tigl3wrapper")

            mgr = _call_first(
                cfg_mod,
                [
                    "CCPACSConfigurationManager_get_instance",
                    "CCPACSConfigurationManager.get_instance",
                ],
            )
            cpacs_config = mgr.get_configuration(native._handle.value)
            wing = _call_first(cpacs_config, ["get_wing", "getWing"], params.wing_uid)

            # Mirror-symmetric wings model one half → double the reference area to
            # the full planform so target_area_m2 / target_aspect_ratio match
            # aviary's convention (agent passes aviary's AREA / AR directly).
            sym_factor = 2.0 if component.symmetry else 1.0

            before = _measure(native, params.wing_uid, component.index, sym_factor)
            cur_span = float(before["span"])
            cur_area = float(before["reference_area"])
            cur_ar = (cur_span**2) / cur_area if cur_area else 0.0

            # Target area / AR default to current if not requested.
            target_area = params.target_area_m2 if params.target_area_m2 is not None else cur_area
            target_ar = (
                params.target_aspect_ratio
                if params.target_aspect_ratio is not None
                else cur_ar
            )

            # span ∝ y-scale, reference_area ∝ x·y. Compound onto current scaling.
            target_span = math.sqrt(target_ar * target_area)
            span_ratio = target_span / cur_span if cur_span else 1.0
            area_ratio = target_area / cur_area if cur_area else 1.0
            y_ratio = span_ratio
            x_ratio = area_ratio / span_ratio if span_ratio else 1.0
            z_ratio = x_ratio  # keep airfoil thickness/chord ratio

            transformation = _call_first(wing, ["get_transformation", "getTransformation"])
            cur = _call_first(transformation, ["get_scaling", "getScaling"])
            new_scaling = geom_mod.CTiglPoint(
                cur.x * x_ratio, cur.y * y_ratio, cur.z * z_ratio
            )
            _call_first(transformation, ["set_scaling", "setScaling"], new_scaling)

            if params.target_sweep_deg is not None:
                _apply_sweep(wing, params.target_sweep_deg)

            try:
                _call_first(cpacs_config, ["invalidate", "Invalidate"])
            except AttributeError:
                pass

            # Persist to the TiXI document, then reopen a fresh handle so the
            # C-API area cache is cleared and get_wing_summary stays consistent.
            uid = _call_first(cpacs_config, ["get_uid", "GetUID"])
            _call_first(cpacs_config, ["write_cpacs", "WriteCPACS"], uid)

            rebuilt = True
            try:
                new_native = tigl_wrapper.Tigl3()
                new_native.open(tixi_handle._tixi_handle, "")
                try:
                    native.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                tigl_handle._tigl_handle = new_native
                native = new_native
            except Exception as exc:  # noqa: BLE001
                # Geometry on the old handle already reflects the morph for span
                # and BREP export; only the cached reference_area read-back is
                # stale. Report that rather than failing outright.
                rebuilt = False
                after = _measure(native, params.wing_uid, component.index, sym_factor)
                return {
                    "wing_uid": component.uid,
                    "requested": requested,
                    "before": before,
                    "after": after,
                    "rebuilt": rebuilt,
                    "note": f"morph applied but reopen failed ({exc}); "
                    "reference_area read-back may be cached.",
                }

            after = _measure(native, params.wing_uid, component.index, sym_factor)

            # Keep the Python-side parameter dict consistent with the geometry.
            component.parameters["span"] = after["span"]
            component.parameters["area"] = after["reference_area"]
            if after.get("aspect_ratio") is not None:
                component.parameters["aspect_ratio"] = after["aspect_ratio"]
            if after.get("sweep_deg") is not None:
                component.parameters["sweep"] = after["sweep_deg"]

            return {
                "wing_uid": component.uid,
                "requested": requested,
                "before": before,
                "after": after,
                "rebuilt": rebuilt,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("MorphError", "Failed to morph wing geometry", str(exc))

    return ToolDefinition(
        name="morph_wing",
        description=(
            "Deform a wing to hit a target reference area, aspect ratio, and/or "
            "sweep, rebuilding the TiGL geometry so downstream meshing and CFD "
            "reflect the change. Returns the MEASURED geometry read back from a "
            "freshly reopened TiGL handle (span, reference_area, aspect_ratio, "
            "sweep, MAC), not just the requested targets."
        ),
        parameters_model=MorphWingParams,
        handler=handler,
        output_schema={},
    )


def morph_fuselage_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the morph_fuselage tool — deform a fuselage and rebuild geometry.

    Length scales the x-axis of the fuselage transformation; diameter scales y
    and z together. Current dimensions and the read-back are measured from the
    exported BREP's bounding box (segfault-free), never the native fuselage
    geometry queries. Like morph_wing it persists via WriteCPACS and reopens a
    fresh handle so downstream tools see the new shape.
    """

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = MorphFuselageParams.model_validate(raw_params)
            tixi_handle, tigl_handle, config = require_session(
                session_manager, params.session_id
            )
            component = config.find_component(params.fuselage_uid)
            if component is None:
                raise_mcp_error("NotFound", f"Fuselage '{params.fuselage_uid}' not found")

            requested = {
                "target_length_m": params.target_length_m,
                "target_diameter_m": params.target_diameter_m,
            }
            if all(v is None for v in requested.values()):
                raise_mcp_error(
                    "MorphError",
                    "Provide at least one of target_length_m, target_diameter_m.",
                )

            native = tigl_handle._tigl_handle

            if native is None:
                for key, value in requested.items():
                    if value is not None:
                        component.parameters[key.replace("target_", "").replace("_m", "")] = value
                return {
                    "fuselage_uid": component.uid,
                    "requested": requested,
                    "rebuilt": False,
                    "note": "No native TiGL runtime — recorded intent only; "
                    "geometry not deformed.",
                }

            cfg_mod = import_module("tigl3.configuration")
            geom_mod = import_module("tigl3.geometry")
            tigl_wrapper = import_module("tigl3.tigl3wrapper")

            mgr = _call_first(
                cfg_mod,
                [
                    "CCPACSConfigurationManager_get_instance",
                    "CCPACSConfigurationManager.get_instance",
                ],
            )
            cpacs_config = mgr.get_configuration(native._handle.value)
            fuselage = _call_first(
                cpacs_config, ["get_fuselage", "getFuselage"], params.fuselage_uid
            )

            before = _fuselage_dims(native, params.fuselage_uid)
            cur_len = before["length"]
            cur_dia = (
                max(before["width"], before["height"])
                if before["width"] is not None and before["height"] is not None
                else None
            )
            if cur_len is None or cur_dia is None:
                raise_mcp_error(
                    "MorphError",
                    "Could not measure current fuselage dimensions (OpenCASCADE "
                    "unavailable); cannot compute the morph.",
                )

            x_ratio = params.target_length_m / cur_len if params.target_length_m else 1.0
            d_ratio = params.target_diameter_m / cur_dia if params.target_diameter_m else 1.0

            transformation = _call_first(
                fuselage, ["get_transformation", "getTransformation"]
            )
            cur = _call_first(transformation, ["get_scaling", "getScaling"])
            new_scaling = geom_mod.CTiglPoint(
                cur.x * x_ratio, cur.y * d_ratio, cur.z * d_ratio
            )
            _call_first(transformation, ["set_scaling", "setScaling"], new_scaling)

            try:
                _call_first(cpacs_config, ["invalidate", "Invalidate"])
            except AttributeError:
                pass

            uid = _call_first(cpacs_config, ["get_uid", "GetUID"])
            _call_first(cpacs_config, ["write_cpacs", "WriteCPACS"], uid)

            rebuilt = True
            try:
                new_native = tigl_wrapper.Tigl3()
                new_native.open(tixi_handle._tixi_handle, "")
                try:
                    native.close()
                except Exception:  # noqa: BLE001
                    pass
                tigl_handle._tigl_handle = new_native
                native = new_native
            except Exception as exc:  # noqa: BLE001
                rebuilt = False
                after = _fuselage_dims(native, params.fuselage_uid)
                return {
                    "fuselage_uid": component.uid,
                    "requested": requested,
                    "before": before,
                    "after": after,
                    "rebuilt": rebuilt,
                    "note": f"morph applied but reopen failed ({exc}).",
                }

            after = _fuselage_dims(native, params.fuselage_uid)
            if after.get("length") is not None:
                component.parameters["length"] = after["length"]
            if after.get("width") is not None and after.get("height") is not None:
                component.parameters["max_diameter"] = max(after["width"], after["height"])

            return {
                "fuselage_uid": component.uid,
                "requested": requested,
                "before": before,
                "after": after,
                "rebuilt": rebuilt,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("MorphError", "Failed to morph fuselage geometry", str(exc))

    return ToolDefinition(
        name="morph_fuselage",
        description=(
            "Deform a fuselage to hit a target length and/or diameter, rebuilding "
            "the TiGL geometry so downstream meshing and CFD reflect the change. "
            "Dimensions are measured from the exported BREP's bounding box "
            "(the native fuselage geometry queries segfault on some CPACS files, "
            "so they are avoided). Returns the MEASURED length/width/height."
        ),
        parameters_model=MorphFuselageParams,
        handler=handler,
        output_schema={},
    )
