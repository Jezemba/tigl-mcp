"""Tools that compute simplified geometric metrics.

When real TiGL bindings are available the tools use native API calls
(``wingGetSpan``, ``wingGetReferenceArea``, etc.) for accurate results.
Otherwise they fall back to the lightweight stub calculations.
"""

from __future__ import annotations

from typing import Any

from tigl_mcp.cpacs import ComponentDefinition, CPACSConfiguration, TiglConfiguration
from tigl_mcp.errors import MCPError, raise_component_not_found, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session


class WingSummaryParams(ToolParameters):
    """Parameters for get_wing_summary."""

    session_id: str
    wing_uid: str


class FuselageSummaryParams(ToolParameters):
    """Parameters for get_fuselage_summary."""

    session_id: str
    fuselage_uid: str


def _safe_get_component(
    config: CPACSConfiguration, uid: str, type_name: str
) -> ComponentDefinition:
    """Resolve a component or raise an MCP error."""
    component = config.find_component(uid)
    if component is None:
        raise_component_not_found(config, uid, type_name)
    return component


# ---------------------------------------------------------------------------
# Real TiGL metric helpers (only called when _tigl_handle is not None)
# ---------------------------------------------------------------------------


def _get_wing_summary_real(  # pragma: no cover
    tigl_handle: TiglConfiguration,
    component: ComponentDefinition,
    wing_uid: str,
) -> dict[str, object]:
    """Compute wing summary via the native TiGL C-API."""
    tigl: Any = tigl_handle._tigl_handle

    span: float = tigl.wingGetSpan(wing_uid)
    half_span = span / 2.0

    # TIGL_NO_SYMMETRY = 0
    reference_area: float = tigl.wingGetReferenceArea(component.index, 0)

    mac_chord: float
    mac_x: float
    mac_y: float
    mac_z: float
    mac_chord, mac_x, mac_y, mac_z = tigl.wingGetMAC(wing_uid)

    top_area = reference_area * 0.5 if reference_area else None
    aspect_ratio = (
        (span**2) / reference_area
        if reference_area and reference_area > 0
        else None
    )

    try:
        wetted_area: float | None = tigl.wingGetWettedArea(wing_uid)
    except Exception:  # noqa: BLE001
        wetted_area = None

    try:
        sweep: float | None = tigl.wingGetSweep(wing_uid)
    except Exception:  # noqa: BLE001
        sweep = component.parameters.get("sweep")

    try:
        dihedral: float | None = tigl.wingGetDihedral(wing_uid)
    except Exception:  # noqa: BLE001
        dihedral = component.parameters.get("dihedral")

    return {
        "span": span,
        "half_span": half_span,
        "reference_area": reference_area,
        "wetted_area": wetted_area,
        "top_area": top_area,
        "aspect_ratio": aspect_ratio,
        "mac_length": mac_chord,
        "mac_quarter_chord": {"x": mac_x, "y": mac_y, "z": mac_z},
        "sweep_deg": sweep,
        "dihedral_deg": dihedral,
        "symmetry": component.symmetry,
    }


def _get_fuselage_summary_real(  # pragma: no cover
    tigl_handle: TiglConfiguration,
    component: ComponentDefinition,
) -> dict[str, object]:
    """Compute fuselage summary via the native TiGL C-API."""
    tigl: Any = tigl_handle._tigl_handle
    idx = component.index

    try:
        volume: float | None = tigl.fuselageGetVolume(idx)
    except Exception:  # noqa: BLE001
        volume = None

    try:
        length: float | None = tigl.fuselageGetCenterLineLength(idx)
    except Exception:  # noqa: BLE001
        length = component.parameters.get("length", 15.0 + idx)

    try:
        wetted_area: float | None = tigl.fuselageGetWettedArea(idx)
    except Exception:  # noqa: BLE001
        wetted_area = None

    try:
        w: float = tigl.fuselageGetMaximalWidth(idx)
        h: float = tigl.fuselageGetMaximalHeight(idx)
        max_cross_section_area: float | None = w * h * 3.14159 / 4.0
    except Exception:  # noqa: BLE001
        max_cross_section_area = None

    try:
        max_diameter: float | None = max(
            tigl.fuselageGetMaximalWidth(idx),
            tigl.fuselageGetMaximalHeight(idx),
        )
    except Exception:  # noqa: BLE001
        max_diameter = None

    return {
        "length": length,
        "wetted_area": wetted_area,
        "max_cross_section_area": max_cross_section_area,
        "max_diameter": max_diameter,
        "approx_volume": volume,
    }


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def get_wing_summary_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_wing_summary tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = WingSummaryParams.model_validate(raw_params)
            _, tigl_handle, config = require_session(session_manager, params.session_id)
            component = _safe_get_component(config, params.wing_uid, "Wing")

            # Try real TiGL first
            if tigl_handle._tigl_handle is not None:  # pragma: no cover
                try:
                    return _get_wing_summary_real(
                        tigl_handle, component, params.wing_uid
                    )
                except Exception:  # noqa: BLE001 - fall through to stub
                    pass

            # Stub / fallback calculations
            span = component.parameters.get("span", 20.0 + component.index)
            reference_area = component.parameters.get("area", span * 0.8)
            half_span = span / 2.0
            top_area = reference_area * 0.5 if reference_area else None
            aspect_ratio = (span**2) / reference_area if reference_area else None
            mac_length = component.parameters.get("mac_length")
            sweep = component.parameters.get("sweep")
            dihedral = component.parameters.get("dihedral")
            mac_quarter_chord = {
                "x": component.bounding_box.xmin
                + 0.25 * (component.bounding_box.xmax - component.bounding_box.xmin),
                "y": component.bounding_box.ymin
                + 0.25 * (component.bounding_box.ymax - component.bounding_box.ymin),
                "z": component.bounding_box.zmin
                + 0.25 * (component.bounding_box.zmax - component.bounding_box.zmin),
            }
            return {
                "span": span,
                "half_span": half_span,
                "reference_area": reference_area,
                "wetted_area": component.parameters.get("wetted_area"),
                "top_area": top_area,
                "aspect_ratio": aspect_ratio,
                "mac_length": mac_length,
                "mac_quarter_chord": mac_quarter_chord,
                "sweep_deg": sweep,
                "dihedral_deg": dihedral,
                "symmetry": component.symmetry,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "WingSummaryError", "Failed to compute wing summary", str(exc)
            )

    return ToolDefinition(
        name="get_wing_summary",
        description="Return key geometric metrics for a wing.",
        parameters_model=WingSummaryParams,
        handler=handler,
        output_schema={},
    )


def get_fuselage_summary_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_fuselage_summary tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = FuselageSummaryParams.model_validate(raw_params)
            _, tigl_handle, config = require_session(session_manager, params.session_id)
            component = _safe_get_component(config, params.fuselage_uid, "Fuselage")

            # NOTE: Real TiGL fuselage calls (fuselageGetVolume, etc.)
            # segfault on some CPACS files (e.g. D150_simple.xml) — the
            # C++ crash kills the server process before Python's try/except
            # can catch it.  Always use stub calculations for fuselage
            # until the upstream TiGL bug is resolved.

            # Stub / fallback calculations
            length = component.parameters.get("length", 15.0 + component.index)
            wetted_area = component.parameters.get("wetted_area")
            max_cross_section_area = component.parameters.get("max_cross_section_area")
            max_diameter = component.parameters.get("max_diameter")
            approx_volume = component.parameters.get("volume")
            return {
                "length": length,
                "wetted_area": wetted_area,
                "max_cross_section_area": max_cross_section_area,
                "max_diameter": max_diameter,
                "approx_volume": approx_volume,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "FuselageSummaryError", "Failed to compute fuselage summary", str(exc)
            )

    return ToolDefinition(
        name="get_fuselage_summary",
        description="Return key geometric metrics for a fuselage.",
        parameters_model=FuselageSummaryParams,
        handler=handler,
        output_schema={},
    )
