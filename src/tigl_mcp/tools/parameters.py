"""Parameter inspection and update tools."""

from __future__ import annotations

from typing import Any

from tigl_mcp.errors import MCPError, raise_component_not_found, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session
from tigl_mcp.tools.morph import apply_fuselage_morph, apply_wing_morph


def _num(value: object) -> float | None:
    """Best-effort float, else None."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _wing_targets(params: dict[str, float]) -> tuple[float | None, float | None, float | None]:
    """Derive (full-planform area, aspect ratio, sweep) from the recorded wing
    parameters. Prefer explicit area/aspect_ratio; otherwise compute the
    trapezoidal planform area from span + root/tip chord."""
    area = _num(params.get("area"))
    ar = _num(params.get("aspect_ratio"))
    span = _num(params.get("span"))
    cr = _num(params.get("root_chord"))
    ct = _num(params.get("tip_chord"))
    sweep = _num(params.get("sweep"))
    if area is None and span is not None and cr is not None and ct is not None:
        area = span * (cr + ct) / 2.0
    if ar is None and span is not None and area:
        ar = (span**2) / area
    return area, ar, sweep


def _fuselage_targets(params: dict[str, float]) -> tuple[float | None, float | None]:
    """Derive (length, diameter) from recorded fuselage parameters."""
    length = _num(params.get("length"))
    diameter = _num(params.get("diameter")) or _num(params.get("max_diameter"))
    if diameter is None:
        radius = _num(params.get("section_radius")) or _num(params.get("radius"))
        if radius is not None:
            diameter = 2.0 * radius
    return length, diameter


# Keys the morphs actually read (see _wing_targets / _fuselage_targets).
_CHORD_KEYS = ("span", "root_chord", "tip_chord")
_PRIMARY_KEYS = {"wing": {"area", "aspect_ratio", "sweep"}, "fuselage": {"length", "diameter"}}
_GEOMETRY_KEYS = {
    "wing": _PRIMARY_KEYS["wing"] | set(_CHORD_KEYS),
    "fuselage": _PRIMARY_KEYS["fuselage"] | {"max_diameter", "radius", "section_radius"},
}


class GetParametersParams(ToolParameters):
    """Parameters for get_high_level_parameters."""

    session_id: str
    component_uid: str


class SetParametersParams(ToolParameters):
    """Parameters for set_high_level_parameters."""

    session_id: str
    component_uid: str
    updates: dict[str, float | str]


def _apply_update(current: float | None, update_value: float | str) -> float:
    """Apply an update string or numeric value to a parameter."""
    if isinstance(update_value, (int, float)):
        return float(update_value)
    if isinstance(update_value, str):
        if update_value.endswith("%"):
            if current is None:
                raise_mcp_error(
                    "UpdateError", "Cannot apply percentage to unknown value"
                )
            delta = float(update_value.rstrip("%")) / 100.0
            return current * (1.0 + delta)
        if update_value.startswith(("+", "-")):
            if current is None:
                raise_mcp_error(
                    "UpdateError", "Cannot apply relative change to unknown value"
                )
            return current + float(update_value)
        return float(update_value)
    raise_mcp_error("UpdateError", "Unsupported update type")


_MEASURED_NOTE = {
    "wing": ("Measured from the current geometry. area, aspect_ratio and span are the full wing (both halves). "
             "sweep is the measured quarter-chord sweep, the value set_high_level_parameters reports after a "
             "reshape; the sweep you pass to set_high_level_parameters sets the sections' positioning angle, "
             "so the measured value differs from it."),
    "fuselage": "Measured from the current geometry (bounding box). diameter is the larger of max_width and max_height.",
}


def _measured_parameters(tigl_handle: Any, component: Any) -> dict[str, float] | None:
    """The component's current geometry, measured, under the keys set_high_level_parameters takes (Avion B145).

    A real CPACS component carries no numeric XML attributes, so ``component.parameters`` was
    empty for every D150 component until something was set -- the tool returned
    ``{"parameters": {}}`` for Wing1, Wing2H, Wing3V and Fuselage1. Uses the same measurements
    as set_high_level_parameters' geometry_morph (wing) and get_fuselage_summary (fuselage).
    Returns None without a real TiGL handle (stub sessions keep the recorded values).
    """
    native = getattr(tigl_handle, "_tigl_handle", None)
    if native is None:
        return None
    type_name = (component.type_name or "").lower()
    try:  # pragma: no cover - needs TiGL
        if type_name == "wing":
            from tigl_mcp.tools.morph import _measure

            m = _measure(native, component.uid, component.index, 2.0 if component.symmetry else 1.0)
            out = {"area": m.get("reference_area"), "aspect_ratio": m.get("aspect_ratio"),
                   "span": m.get("span"), "sweep": m.get("sweep_deg"), "taper_ratio": m.get("taper_ratio")}
        elif type_name == "fuselage":
            from tigl_mcp.tools.morph import _fuselage_dims

            d = _fuselage_dims(native, component.uid)
            w, h = d.get("width"), d.get("height")
            out = {"length": d.get("length"), "max_width": w, "max_height": h,
                   "diameter": max(w, h) if w is not None and h is not None else None}
        else:
            return None
    except Exception:  # noqa: BLE001 - fall back to the recorded values rather than fail the call
        return None
    out = {k: float(v) for k, v in out.items() if v is not None}
    return out or None


def get_high_level_parameters_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_high_level_parameters tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = GetParametersParams.model_validate(raw_params)
            _, tigl_handle, config = require_session(session_manager, params.session_id)
            component = config.find_component(params.component_uid)
            if component is None:
                raise_component_not_found(config, params.component_uid)
            measured = _measured_parameters(tigl_handle, component)
            if measured is None:
                return {"component_uid": component.uid, "parameters": component.parameters}
            return {"component_uid": component.uid, "parameters": {**component.parameters, **measured},
                    "measured": True, "note": _MEASURED_NOTE.get((component.type_name or "").lower(), "")}
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("ParameterError", "Failed to fetch parameters", str(exc))

    return ToolDefinition(
        name="get_high_level_parameters",
        description=(
            "Return a component's current high-level parameters, measured from the geometry: for a wing "
            "area, aspect_ratio and span (full wing), sweep (quarter-chord) and taper_ratio; for a fuselage "
            "length, diameter, max_width and max_height."
        ),
        parameters_model=GetParametersParams,
        handler=handler,
        output_schema={},
    )


def set_high_level_parameters_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the set_high_level_parameters tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = SetParametersParams.model_validate(raw_params)
            tixi_handle, tigl_handle, config = require_session(
                session_manager, params.session_id
            )
            component = config.find_component(params.component_uid)
            if component is None:
                raise_component_not_found(config, params.component_uid)
            warnings: list[str] = []
            for key, value in params.updates.items():
                current_value = component.parameters.get(key)
                try:
                    component.parameters[key] = _apply_update(current_value, value)
                except MCPError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive path
                    warnings.append(f"Skipped '{key}': {exc}")

            # Actually DEFORM the geometry (not just record intent). When a real
            # TiGL handle is present and this is a wing/fuselage, translate the
            # recorded parameters into a morph + rebuild so the mesh/CFD reflect
            # them. This closes the aero<->geometry loop without any change to
            # the agents — they already call set_high_level_parameters.
            morph_result: dict[str, object] | None = None
            type_name = (component.type_name or "").lower()
            if tigl_handle._tigl_handle is not None:
                try:
                    if type_name == "wing":
                        area, ar, sweep = _wing_targets(component.parameters)
                        if any(v is not None for v in (area, ar, sweep)):
                            morph_result = apply_wing_morph(
                                tixi_handle, tigl_handle, component,
                                target_area_m2=area,
                                target_aspect_ratio=ar,
                                target_sweep_deg=sweep,
                            )
                    elif type_name == "fuselage":
                        length, diameter = _fuselage_targets(component.parameters)
                        if length is not None or diameter is not None:
                            morph_result = apply_fuselage_morph(
                                tixi_handle, tigl_handle, component,
                                target_length_m=length,
                                target_diameter_m=diameter,
                            )
                except MCPError as exc:
                    warnings.append(f"Geometry morph skipped: {exc.error}")
                except Exception as exc:  # noqa: BLE001 - never fail the record on morph
                    warnings.append(f"Geometry morph failed: {exc}")

            # B105: say which keys did NOT move the geometry instead of succeeding silently. The wing
            # morph reads area / aspect_ratio / sweep, or span + root_chord + tip_chord together
            # (area only); the fuselage morph reads length and a diameter (or radius).
            geometry_keys = _GEOMETRY_KEYS.get(type_name)
            if geometry_keys is not None:
                ignored = sorted(k for k in params.updates if k not in geometry_keys)
                if type_name == "wing" and any(k in params.updates for k in _CHORD_KEYS) and not all(
                    k in component.parameters for k in _CHORD_KEYS
                ):
                    ignored = sorted(set(ignored) | {k for k in params.updates if k in _CHORD_KEYS})
                if ignored:
                    warnings.append(
                        f"Ignored (not a geometry input for a {type_name}): {', '.join(ignored)}. "
                        f"Inputs that reshape it: {', '.join(sorted(_PRIMARY_KEYS[type_name]))}."
                    )

            result: dict[str, object] = {
                "component_uid": component.uid,
                "new_parameters": component.parameters,
                "warnings": warnings,
            }
            if morph_result is not None:
                result["geometry_morph"] = {
                    "rebuilt": morph_result.get("rebuilt"),
                    "before": morph_result.get("before"),
                    "after": morph_result.get("after"),
                }
            return result
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "ParameterError", "Failed to apply parameter updates", str(exc)
            )

    return ToolDefinition(
        name="set_high_level_parameters",
        description=(
            "Update high-level design parameters for a component. For a wing or "
            "fuselage with a live TiGL runtime this now DEFORMS the geometry and "
            "rebuilds it (wing: area/AR from span+chords, plus sweep; fuselage: "
            "length + diameter), so downstream meshing and CFD reflect the change "
            "— the returned geometry_morph.after holds the measured new geometry."
        ),
        parameters_model=SetParametersParams,
        handler=handler,
        output_schema={},
    )
