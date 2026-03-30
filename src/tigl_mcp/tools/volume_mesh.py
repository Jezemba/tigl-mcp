"""Volume mesh generation tool using gmsh for CFD applications.

Requires both real TiGL bindings (for STL surface export) and the gmsh
Python API.  When either dependency is missing the tool reports a clear
error rather than returning stub data.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Any, Literal

from tigl_mcp.cpacs import ComponentDefinition, TiglConfiguration
from tigl_mcp.errors import MCPError, raise_mcp_error
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tooling import ToolDefinition, ToolParameters
from tigl_mcp.tools.common import require_session

try:
    import gmsh  # type: ignore[import-untyped]

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


# ---------------------------------------------------------------------------
# Parameter model
# ---------------------------------------------------------------------------


class GenerateVolumeMeshParams(ToolParameters):
    """Parameters for generate_volume_mesh tool."""

    session_id: str
    component_uid: str | None = None  # None means entire configuration
    far_field_distance: float = 10.0
    mesh_size_min: float = 0.1
    mesh_size_max: float = 5.0
    surface_mesh_size: float = 0.5
    boundary_layer_enabled: bool = False
    boundary_layer_thickness: float = 0.01
    boundary_layer_layers: int = 5
    boundary_layer_growth: float = 1.2
    output_format: Literal["su2", "msh"] = "su2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_stl_for_component(
    tigl_handle: TiglConfiguration, component: ComponentDefinition
) -> bytes:
    """Export component surface as STL bytes using TiGL."""
    tigl: Any = tigl_handle._tigl_handle
    if tigl is None:
        raise_mcp_error("ExportError", "TiGL handle not available")

    deflection = 0.01
    comp_type = component.type_name.lower()

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if comp_type == "wing":
            tigl.exportMeshedWingSTL(component.index, tmp_path, deflection)
        elif comp_type == "fuselage":
            tigl.exportMeshedFuselageSTL(component.index, tmp_path, deflection)
        else:
            raise_mcp_error(
                "ExportError",
                f"STL export not supported for component type {comp_type}",
            )
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:  # noqa: BLE001
            pass


def _export_configuration_stl(tigl_handle: TiglConfiguration) -> bytes:
    """Export entire configuration as a single STL file."""
    tigl: Any = tigl_handle._tigl_handle
    if tigl is None:
        raise_mcp_error("ExportError", "TiGL handle not available")

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tigl.exportMeshedGeometrySTL(tmp_path, 0.01)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    except Exception:
        # Fallback: export each component individually and concatenate.
        all_stl = b""
        config = tigl_handle.cpacs_configuration
        for wing in config.wings:
            all_stl += _export_stl_for_component(tigl_handle, wing)
        for fuselage in config.fuselages:
            all_stl += _export_stl_for_component(tigl_handle, fuselage)
        return all_stl
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:  # noqa: BLE001
            pass


def _get_stl_bounding_box(
    stl_path: str,
) -> tuple[float, float, float, float, float, float]:
    """Parse an ASCII STL file to determine its axis-aligned bounding box."""
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    with open(stl_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("vertex"):
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        min_z, max_z = min(min_z, z), max(max_z, z)
                    except ValueError:
                        continue

    if min_x == float("inf"):
        raise_mcp_error(
            "MeshError", "Could not parse STL bounding box - no vertices found"
        )

    return min_x, max_x, min_y, max_y, min_z, max_z


def _generate_volume_mesh_gmsh(
    stl_bytes: bytes,
    params: GenerateVolumeMeshParams,
) -> tuple[bytes, dict[str, Any]]:
    """Generate a volume mesh using gmsh with the STL embedded in a far-field box."""
    if not HAS_GMSH:
        raise_mcp_error("DependencyError", "gmsh is not installed or not available")

    with tempfile.NamedTemporaryFile(
        suffix=".stl", delete=False, mode="wb"
    ) as stl_file:
        stl_file.write(stl_bytes)
        stl_path = stl_file.name

    output_suffix = ".su2" if params.output_format == "su2" else ".msh"
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as out_file:
        output_path = out_file.name

    stats: dict[str, Any] = {}

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volume_mesh")

        # Determine domain size from the STL bounding box
        min_x, max_x, min_y, max_y, min_z, max_z = _get_stl_bounding_box(stl_path)
        char_length = max(max_x - min_x, max_y - min_y, max_z - min_z)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        cz = (min_z + max_z) / 2
        ff = params.far_field_distance * char_length

        stats["characteristic_length"] = char_length
        stats["domain_size"] = {
            "x": [cx - ff, cx + ff],
            "y": [cy - ff, cy + ff],
            "z": [cz - ff, cz + ff],
        }

        # Create far-field box (OCC kernel)
        box_tag = gmsh.model.occ.addBox(
            cx - ff, cy - ff, cz - ff, 2 * ff, 2 * ff, 2 * ff
        )
        gmsh.model.occ.synchronize()

        # Import the aircraft STL
        gmsh.merge(stl_path)

        # Separate box surfaces (tags 1-6) from aircraft surfaces
        all_surfaces = gmsh.model.getEntities(2)
        box_surface_tags = [1, 2, 3, 4, 5, 6]
        ac_tags = [tag for _, tag in all_surfaces if tag not in box_surface_tags]

        # Classify discrete STL mesh into surfaces if needed
        if not ac_tags:
            try:
                angle_rad = 40 * 3.14159 / 180
                gmsh.model.mesh.classifySurfaces(
                    angle_rad, True, True, 180 * 3.14159 / 180
                )
                gmsh.model.mesh.createGeometry()
                all_surfaces = gmsh.model.getEntities(2)
                ac_tags = [
                    tag for _, tag in all_surfaces if tag not in box_surface_tags
                ]
            except Exception:  # noqa: BLE001
                pass

        # Reverse normals on aircraft surfaces so they point inward
        if ac_tags:
            gmsh.model.mesh.reverse([(2, t) for t in ac_tags])

        # Embed aircraft surfaces into the box volume
        if ac_tags:
            gmsh.model.mesh.embed(2, ac_tags, 3, box_tag)

        # Mesh size options
        gmsh.option.setNumber("Mesh.MeshSizeMin", params.mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", params.mesh_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Physical groups: farfield, aircraft, fluid volume
        existing = [s[1] for s in gmsh.model.getEntities(2)]
        valid_box = [t for t in box_surface_tags if t in existing]
        if valid_box:
            fg = gmsh.model.addPhysicalGroup(2, valid_box)
            gmsh.model.setPhysicalName(2, fg, "farfield")
        stats["farfield_surfaces"] = len(valid_box)

        valid_ac = [t for t in ac_tags if t in existing]
        if valid_ac:
            ag = gmsh.model.addPhysicalGroup(2, valid_ac)
            gmsh.model.setPhysicalName(2, ag, "aircraft")
        stats["aircraft_surfaces"] = len(valid_ac)

        all_volumes = gmsh.model.getEntities(3)
        if all_volumes:
            vol_tags = [v[1] for v in all_volumes]
            vg = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, vg, "fluid")

        # Statistics
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["node_count"] = len(node_tags)
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        stats["element_count"] = sum(len(tags) for tags in elem_tags)

        # Write output
        if params.output_format == "su2":
            gmsh.option.setNumber("Mesh.Format", 42)
        else:
            gmsh.option.setNumber("Mesh.Format", 1)
        gmsh.write(output_path)

        with open(output_path, "rb") as fh:
            mesh_bytes = fh.read()

        return mesh_bytes, stats

    except MCPError:
        raise
    except Exception as exc:
        raise_mcp_error("MeshError", f"gmsh meshing failed: {exc}")
    finally:
        try:
            gmsh.finalize()
        except Exception:  # noqa: BLE001
            pass
        for p in (stl_path, output_path):
            try:
                os.unlink(p)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def generate_volume_mesh_tool(session_manager: SessionManager) -> ToolDefinition:
    """Create the generate_volume_mesh tool."""

    def handler(raw_params: dict[str, object]) -> dict[str, object]:
        try:
            params = GenerateVolumeMeshParams.model_validate(raw_params)

            if not HAS_GMSH:
                raise_mcp_error(
                    "DependencyError",
                    "gmsh is not installed. Install with: pip install gmsh",
                )

            _, tigl_handle, config = require_session(
                session_manager, params.session_id
            )

            if tigl_handle._tigl_handle is None:
                raise_mcp_error(
                    "ExportError",
                    "Volume mesh requires real TiGL bindings (not stub mode).",
                )

            # Export STL surface
            if params.component_uid is not None:
                component = config.find_component(params.component_uid)
                if component is None:
                    raise_mcp_error(
                        "NotFound", f"Component '{params.component_uid}' not found"
                    )
                stl_bytes = _export_stl_for_component(tigl_handle, component)
            else:
                stl_bytes = _export_configuration_stl(tigl_handle)

            if not stl_bytes or len(stl_bytes) < 100:
                raise_mcp_error(
                    "ExportError", "Failed to export STL - empty or too small"
                )

            mesh_bytes, stats = _generate_volume_mesh_gmsh(stl_bytes, params)

            if params.output_format == "su2" and b"NDIME=" not in mesh_bytes:
                raise_mcp_error(
                    "MeshError", "Generated SU2 file is invalid - missing NDIME"
                )

            mesh_base64 = base64.b64encode(mesh_bytes).decode("utf-8")
            return {
                "format": params.output_format,
                "mesh_base64": mesh_base64,
                "mesh_size_bytes": len(mesh_bytes),
                "statistics": stats,
            }

        except MCPError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            raise_mcp_error("MeshError", f"Volume mesh generation failed: {exc}")

    return ToolDefinition(
        name="generate_volume_mesh",
        description=(
            "Generate a 3D volume mesh suitable for CFD simulation using gmsh. "
            "Creates a flow domain around the aircraft geometry with proper "
            "boundary markers for SU2 or other CFD solvers."
        ),
        parameters_model=GenerateVolumeMeshParams,
        handler=handler,
        output_schema={
            "format": "string",
            "mesh_base64": "string",
            "mesh_size_bytes": "integer",
            "statistics": "object",
        },
    )
