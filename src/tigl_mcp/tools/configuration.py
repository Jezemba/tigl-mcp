"""Tools for inspecting CPACS configurations."""

from __future__ import annotations

from typing import Any

from tigl_mcp.cpacs import ComponentDefinition
from tigl_mcp.errors import MCPError, raise_component_not_found, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session


class SessionOnlyParams(ToolParameters):
    """Parameters that only require a session identifier."""

    session_id: str


class ListComponentsParams(SessionOnlyParams):
    """Parameters for list_geometric_components."""

    type_filter: str | None = None


class ComponentMetadataParams(SessionOnlyParams):
    """Parameters for get_component_metadata."""

    component_uid: str


# Avion final re-run (2026-11-18): every bounding box here was BoundingBox.from_index -- a box made up from
# the component's index (component 1 spans x 1..2), reported as the aircraft's. With a TiGL runtime the box
# is measured from the exported BREP; otherwise it is reported as null with the reason.
_BBOX_NOT_MEASURED = ("not measured: no TiGL runtime in this server, and the tool does not report placeholder "
                      "boxes")


def _measured_bbox(native: Any, component: ComponentDefinition) -> dict[str, float] | None:
    """Axis-aligned box of one component from its exported BREP, mirrored for a symmetric wing; None on failure."""
    import os
    import tempfile

    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepTools import breptools_Read
        from OCC.Core.TopoDS import TopoDS_Shape
    except Exception:  # noqa: BLE001
        return None
    kind = (component.type_name or "").lower()
    export = {"wing": "exportWingBREPByUID", "fuselage": "exportFuselageBREPByUID"}.get(kind)
    if export is None or not hasattr(native, export):
        return None
    path = tempfile.mktemp(suffix=".brep")
    try:
        getattr(native, export)(component.uid, path)
        shape = TopoDS_Shape()
        breptools_Read(shape, path, BRep_Builder())
        box = Bnd_Box()
        brepbndlib_Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass
    sym = str(component.symmetry or "").lower()
    if kind == "wing" and sym not in ("", "0", "none"):
        # CPACS models one side; the mirror image completes the box.
        if "x-z" in sym or sym == "2":
            ymin, ymax = min(ymin, -ymax), max(ymax, -ymin)
        elif "x-y" in sym or sym == "1":
            zmin, zmax = min(zmin, -zmax), max(zmax, -zmin)
        elif "y-z" in sym or sym == "3":
            xmin, xmax = min(xmin, -xmax), max(xmax, -xmin)
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}


def _component_bbox(tigl_handle: Any, component: ComponentDefinition) -> tuple[dict | None, str]:
    """(box or None, how it was obtained) for one component."""
    native = getattr(tigl_handle, "_tigl_handle", None)
    if native is None:
        return None, _BBOX_NOT_MEASURED
    box = _measured_bbox(native, component)
    if box is None:
        return None, f"not measured: the {component.type_name} BREP could not be exported or read"
    return box, "measured from the exported BREP"


def _combine_boxes(boxes: list[dict]) -> dict[str, float]:
    return {"xmin": min(b["xmin"] for b in boxes), "xmax": max(b["xmax"] for b in boxes),
            "ymin": min(b["ymin"] for b in boxes), "ymax": max(b["ymax"] for b in boxes),
            "zmin": min(b["zmin"] for b in boxes), "zmax": max(b["zmax"] for b in boxes)}


def _segment_counts(tigl_handle: Any, component: ComponentDefinition) -> dict[str, object]:
    """Section/segment counts from TiGL; recorded values (or null) without a runtime -- never a made-up 0."""
    native = getattr(tigl_handle, "_tigl_handle", None)
    kind = (component.type_name or "").lower()

    def ask(name: str) -> int | None:
        try:
            return int(getattr(native, name)(component.index))
        except Exception:  # noqa: BLE001
            return None

    if kind == "wing":
        if native is not None:
            return {"num_sections": ask("wingGetSectionCount"), "num_segments": ask("wingGetSegmentCount"),
                    "num_component_segments": ask("wingGetComponentSegmentCount")}
        return {"num_sections": component.parameters.get("sections"),
                "num_segments": component.parameters.get("segments"),
                "num_component_segments": component.parameters.get("component_segments")}
    if native is not None:
        return {"num_segments": ask("fuselageGetSegmentCount")}
    return {"num_segments": component.parameters.get("segments")}


def _component_to_dict(component: ComponentDefinition) -> dict[str, object]:
    """Convert a component into a JSON-serializable dictionary."""
    return {
        "uid": component.uid,
        "name": component.name,
        "index": component.index,
        "type": component.type_name,
        "parent_uid": None,
        "label": component.name,
        "symmetry": component.symmetry,
        "bounding_box": None,  # never the index-made placeholder box (list_geometric_components drops it)
        "parameters": component.parameters,
    }


def get_configuration_summary_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_configuration_summary tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = SessionOnlyParams.model_validate(raw_params)
            _, tigl_handle, config = require_session(session_manager, params.session_id)
            boxes, failed = [], []
            for comp in [*config.wings, *config.fuselages]:
                box, how = _component_bbox(tigl_handle, comp)
                (boxes.append(box) if box is not None else failed.append(f"{comp.uid}: {how}"))
            if boxes and not failed:
                bounding_box: dict | None = _combine_boxes(boxes)
                bbox_note = ("measured from the exported BREPs of every wing (mirrored where symmetric) and "
                             "fuselage; engines and rotors are not included")
            else:
                bounding_box = None
                bbox_note = ("; ".join(failed) if failed else "no wing or fuselage to measure")
            wings = [
                {"uid": wing.uid, "name": wing.name, "index": wing.index}
                for wing in config.wings
            ]
            fuselages = [
                {"uid": fuselage.uid, "name": fuselage.name, "index": fuselage.index}
                for fuselage in config.fuselages
            ]
            rotors = [
                {"uid": rotor.uid, "name": rotor.name, "index": rotor.index}
                for rotor in config.rotors
            ]
            engines = [
                {"uid": engine.uid, "name": engine.name, "index": engine.index}
                for engine in config.engines
            ]
            return {
                "wings": wings,
                "fuselages": fuselages,
                "rotors": rotors,
                "engines": engines,
                "bounding_box": bounding_box,
                "bounding_box_note": bbox_note,
            }
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "SummaryError", "Failed to build configuration summary", str(exc)
            )

    return ToolDefinition(
        name="get_configuration_summary",
        description="Return component lists and the overall bounding box.",
        parameters_model=SessionOnlyParams,
        handler=handler,
        output_schema={},
    )


def list_geometric_components_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the list_geometric_components tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, list[dict[str, object]]]:
        try:
            params = ListComponentsParams.model_validate(raw_params)
            _, _, config = require_session(session_manager, params.session_id)
            components: list[dict[str, object]] = [
                _component_to_dict(component) for component in config.all_components()
            ]
            if params.type_filter:
                components = [
                    component
                    for component in components
                    if str(component["type"]).lower() == params.type_filter.lower()
                ]
            for component in components:
                component.pop("parameters", None)
                component.pop("symmetry", None)
                component.pop("bounding_box", None)
            return {"components": components}
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("ListError", "Failed to list components", str(exc))

    return ToolDefinition(
        name="list_geometric_components",
        description="List CPACS geometric components.",
        parameters_model=ListComponentsParams,
        handler=handler,
        output_schema={},
    )


def get_component_metadata_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the get_component_metadata tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = ComponentMetadataParams.model_validate(raw_params)
            _, tigl_handle, config = require_session(session_manager, params.session_id)
            component = config.find_component(params.component_uid)
            if component is None:
                raise_component_not_found(config, params.component_uid)
            box, how = _component_bbox(tigl_handle, component)
            counts = _segment_counts(tigl_handle, component)
            kind = component.type_name.lower()
            metadata: dict[str, object] = {
                "uid": component.uid,
                "type": component.type_name,
                # Not read from the CPACS tree: unknown, not "none".
                "parent_uid": None,
                "children_uids": None,
                "symmetry": component.symmetry,
                "bounding_box": box,
                "bounding_box_note": how,
                "wing_data": counts if kind == "wing" else None,
                "fuselage_data": counts if kind == "fuselage" else None,
                "note": ("parent_uid and children_uids are not read from the CPACS file (null = unknown). "
                         + ("Counts are from TiGL." if getattr(tigl_handle, "_tigl_handle", None) is not None
                            else "No TiGL runtime: counts are recorded values, null when not recorded.")),
            }
            return metadata
        except MCPError as error:
            raise error
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error(
                "MetadataError", "Failed to read component metadata", str(exc)
            )

    return ToolDefinition(
        name="get_component_metadata",
        description="Return metadata for a geometric component.",
        parameters_model=ComponentMetadataParams,
        handler=handler,
        output_schema={},
    )
