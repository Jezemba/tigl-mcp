"""Parameter inspection and update tools."""

from __future__ import annotations

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


def get_high_level_parameters_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_high_level_parameters tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = GetParametersParams.model_validate(raw_params)
            _, _, config = require_session(session_manager, params.session_id)
            component = config.find_component(params.component_uid)
            if component is None:
                raise_component_not_found(config, params.component_uid)
            return {"component_uid": component.uid, "parameters": component.parameters}
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("ParameterError", "Failed to fetch parameters", str(exc))

    return ToolDefinition(
        name="get_high_level_parameters",
        description="Return high-level design parameters for a component.",
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
