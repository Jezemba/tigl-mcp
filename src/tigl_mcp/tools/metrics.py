"""Tools that report geometric metrics of a wing or a fuselage.

With a TiGL runtime every number is measured from the geometry (TiGL native queries, or the exported
BREP for the fuselage). If a measurement fails the tool says so and reports nothing for it: it never
substitutes a placeholder (Avion final re-run, 2026-11-18: on any TiGL exception the wing summary used to
fall back to span 20 + index, area 0.8 x span, and the fuselage to length 15 + index, unlabelled).

Without a TiGL runtime (unit tests, a server without the bindings) the summaries report only the values
recorded on the component (CPACS attributes or earlier set_high_level_parameters requests), with
``measured: false``; anything not recorded is null.

Area conventions (Avion F5, M13). TiGL models a mirror-symmetric wing as one side:
  * ``reference_area`` is ONE side's projected area (TiGL wingGetReferenceArea, no symmetry): 61.39 m^2
    for the D150 Wing1. ``full_reference_area`` is both sides (122.78 m^2).
  * ``aspect_ratio`` is the FULL wing's, span^2 / full area (9.37 for the D150). It used to be span^2 /
    one-side area (18.73), double the real value; agents read that one.
  * ``wetted_area`` is ONE side's EXPOSED wetted area (TiGL wingGetWettedArea: the wing surface outside the
    fuselage), 111.58 m^2 for the D150 Wing1, whose gross one-side surface (wingGetSurfaceArea) is
    130.51 m^2. ``wetted_area_exposed_full`` is both sides. Use exposed areas with the fuselage's wetted
    area in a drag build-up, so the wing root inside the fuselage is not counted.
  * The fuselage ``wetted_area`` is its whole loft surface (417.07 m^2 for the D150), gross: the wing and
    tail root junctions are NOT subtracted. A fuselage is not mirrored, so this is the whole fuselage.
"""

from __future__ import annotations

from typing import Any

from tigl_mcp.cpacs import ComponentDefinition, CPACSConfiguration, TiglConfiguration
from tigl_mcp.errors import MCPError, raise_component_not_found, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session

WING_AREA_CONVENTIONS = (
    "reference_area and wetted_area are ONE side of a mirror-symmetric wing; full_reference_area and "
    "wetted_area_exposed_full are both sides. aspect_ratio is the full wing's (span^2 / full_reference_area). "
    "wetted_area is EXPOSED (TiGL subtracts the part inside the fuselage); surface_area_gross_* include it. "
    "For a drag build-up add wetted_area_exposed_full to the fuselage wetted_area and divide by "
    "full_reference_area."
)
FUSELAGE_AREA_CONVENTIONS = (
    "wetted_area is the whole fuselage loft surface, measured from its BREP; gross: the wing and tail root "
    "junctions are not subtracted. The fuselage is not mirrored, so this is the whole fuselage."
)
_STUB_NOTE = (
    "No TiGL runtime in this server: these are the values recorded on the component (CPACS attributes or "
    "earlier set_high_level_parameters requests), NOT measurements. Values not recorded are null."
)


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


def _get_wing_summary_real(
    tigl_handle: TiglConfiguration,
    component: ComponentDefinition,
    wing_uid: str,
) -> dict[str, object]:
    """Measure a wing via the native TiGL API. Raises if a core quantity (span, area, MAC) cannot be read."""
    tigl: Any = tigl_handle._tigl_handle
    warnings: list[str] = []

    span: float = tigl.wingGetSpan(wing_uid)
    # TIGL_NO_SYMMETRY = 0: one side of a mirror-symmetric wing.
    reference_area: float = tigl.wingGetReferenceArea(component.index, 0)
    mac_chord, mac_x, mac_y, mac_z = tigl.wingGetMAC(wing_uid)
    sym_factor = 2.0 if component.symmetry else 1.0
    full_area = reference_area * sym_factor if reference_area else None
    full_ar = (span**2) / full_area if full_area else None

    try:
        wetted_area: float | None = float(tigl.wingGetWettedArea(wing_uid))
    except Exception as exc:  # noqa: BLE001
        wetted_area = None
        warnings.append(f"wetted area not measured (TiGL wingGetWettedArea failed: {exc})")
    try:
        gross_area: float | None = float(tigl.wingGetSurfaceArea(component.index))
    except Exception as exc:  # noqa: BLE001
        gross_area = None
        warnings.append(f"gross surface area not measured (TiGL wingGetSurfaceArea failed: {exc})")

    # TiGL has no sweep query; the planform is measured from the section chord points.
    from tigl_mcp.tools.planform import measure_planform

    planform = measure_planform(tigl, component.index)
    if planform is None:
        planform = {}
        warnings.append("sweep and taper not measured (the section chord points could not be read)")

    out: dict[str, object] = {
        "span": span,
        "half_span": span / 2.0,
        "reference_area": reference_area,
        "wetted_area": wetted_area,
        "wetted_area_exposed_one_side": wetted_area,
        "wetted_area_exposed_full": wetted_area * sym_factor if wetted_area is not None else None,
        "surface_area_gross_one_side": gross_area,
        "surface_area_gross_full": gross_area * sym_factor if gross_area is not None else None,
        "aspect_ratio": full_ar,
        "mac_length": mac_chord,
        "mac_quarter_chord": {"x": mac_x, "y": mac_y, "z": mac_z},
        "sweep_deg": planform.get("sweep_quarter_chord_deg"),
        "sweep_le_deg": planform.get("sweep_le_deg"),
        "sweep_quarter_chord_deg": planform.get("sweep_quarter_chord_deg"),
        "taper_ratio": planform.get("taper_ratio"),
        "root_chord_m": planform.get("root_chord_m"),
        "tip_chord_m": planform.get("tip_chord_m"),
        "full_reference_area": full_area,
        "full_aspect_ratio": full_ar,
        # TiGL's Python API has no dihedral query (wingGetDihedral does not exist); a recorded request is
        # not a measurement, so it is not reported as one.
        "dihedral_deg": None,
        "symmetry": component.symmetry,
        "measured": True,
        "area_conventions": WING_AREA_CONVENTIONS,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _stub_values(component: ComponentDefinition, keys: dict[str, str]) -> dict[str, object]:
    """{output key: recorded value or None} -- never a made-up number."""
    return {out: component.parameters.get(src) for out, src in keys.items()}


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

            if tigl_handle._tigl_handle is not None:
                try:
                    return _get_wing_summary_real(tigl_handle, component, params.wing_uid)
                except Exception as exc:  # noqa: BLE001
                    raise_mcp_error(
                        "WingSummaryError",
                        f"Could not measure wing '{component.uid}' with TiGL, so no values are reported "
                        "(the tool does not substitute placeholder numbers).",
                        str(exc),
                    )

            out = _stub_values(component, {
                "span": "span", "reference_area": "area", "wetted_area": "wetted_area",
                "mac_length": "mac_length", "sweep_deg": "sweep", "dihedral_deg": "dihedral"})
            span, area = out["span"], out["reference_area"]
            out["half_span"] = span / 2.0 if isinstance(span, (int, float)) else None
            out["aspect_ratio"] = (span**2 / area if isinstance(span, (int, float))
                                   and isinstance(area, (int, float)) and area > 0 else None)
            out["mac_quarter_chord"] = None
            out["symmetry"] = component.symmetry
            out["measured"] = False
            out["note"] = _STUB_NOTE
            return out
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "WingSummaryError", "Failed to compute wing summary", str(exc)
            )

    return ToolDefinition(
        name="get_wing_summary",
        description=(
            "Return key geometric metrics for a wing, measured from the geometry. reference_area and "
            "wetted_area are ONE side of a mirror-symmetric wing (full_reference_area, "
            "wetted_area_exposed_full are both sides); aspect_ratio is the full wing's; wetted_area is the "
            "exposed area outside the fuselage."
        ),
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

            # NOTE: Real TiGL fuselage calls (fuselageGetVolume, etc.) segfault on some CPACS files (e.g.
            # D150_simple.xml) -- the C++ crash kills the server process before Python's try/except can
            # catch it. Length, width, height and the surface (wetted) area are measured instead from the
            # exported BREP (OpenCASCADE), the same measurement morph_fuselage reports.
            if tigl_handle._tigl_handle is not None:
                from tigl_mcp.tools.morph import _fuselage_dims

                try:
                    dims = _fuselage_dims(tigl_handle._tigl_handle, component.uid)
                except Exception as exc:  # noqa: BLE001
                    dims, error = {}, str(exc)
                else:
                    error = "the BREP export or OpenCASCADE was unavailable"
                if dims.get("length") is None:
                    raise_mcp_error(
                        "FuselageSummaryError",
                        f"Could not measure fuselage '{component.uid}', so no values are reported (the tool "
                        "does not substitute placeholder numbers).",
                        error,
                    )
                out: dict[str, object] = {
                    "length": dims["length"],
                    "max_width": dims["width"],
                    "max_height": dims["height"],
                    "max_diameter": max(dims["width"], dims["height"]),
                    "wetted_area": dims.get("surface_area"),
                    "max_cross_section_area": None,
                    "approx_volume": None,
                    "measured": True,
                    "area_conventions": FUSELAGE_AREA_CONVENTIONS,
                    "not_measured": ["max_cross_section_area", "approx_volume"],
                }
                if dims.get("surface_area") is None:
                    out["warnings"] = ["wetted area not measured (the BREP surface area could not be computed)"]
                return out

            out = _stub_values(component, {
                "length": "length", "wetted_area": "wetted_area",
                "max_cross_section_area": "max_cross_section_area", "max_diameter": "max_diameter",
                "approx_volume": "volume"})
            out["measured"] = False
            out["note"] = _STUB_NOTE
            return out
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "FuselageSummaryError", "Failed to compute fuselage summary", str(exc)
            )

    return ToolDefinition(
        name="get_fuselage_summary",
        description=(
            "Return key geometric metrics for a fuselage, measured from its geometry: length, max width, "
            "height and diameter, and wetted_area (whole fuselage loft surface, gross of the wing and tail "
            "root junctions)."
        ),
        parameters_model=FuselageSummaryParams,
        handler=handler,
        output_schema={},
    )
