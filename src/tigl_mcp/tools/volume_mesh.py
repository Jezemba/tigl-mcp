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


def _export_brep_for_component(
    tigl_handle: TiglConfiguration, component: ComponentDefinition
) -> bytes | None:
    """Export component as native OCC BREP using TiGL.

    Prefer this over STL: the BREP is a parametric watertight CAD solid
    that gmsh's OCC kernel imports directly. The STL path requires
    ``BRepBuilderAPI_Sewing`` to stitch triangulation into a shell, which
    leaks at segment boundaries on swept wings and produces meshes with
    fluid cells inside the wing solid (Phase G.1 bug, 2026-05-18).

    Returns ``None`` if the TiGL build doesn't expose the BREP exporter
    for this component type, so callers can fall back to STL.
    """
    tigl: Any = tigl_handle._tigl_handle
    if tigl is None:
        return None

    comp_type = component.type_name.lower()
    if comp_type == "wing":
        exporter = getattr(tigl, "exportWingBREPByUID", None)
    elif comp_type == "fuselage":
        exporter = getattr(tigl, "exportFuselageBREPByUID", None)
    else:
        return None

    if exporter is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exporter(component.uid, tmp_path)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:  # noqa: BLE001
            pass


def _export_configuration_brep(tigl_handle: TiglConfiguration) -> bytes | None:
    """Export the entire fused configuration as OCC BREP via TiGL.

    Returns ``None`` if exportFusedBREP isn't available, so callers can
    fall back to per-component or STL paths.
    """
    tigl: Any = tigl_handle._tigl_handle
    if tigl is None:
        return None
    exporter = getattr(tigl, "exportFusedBREP", None)
    if exporter is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        exporter(tmp_path)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return None
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


def _stl_bytes_to_solid_brep(stl_bytes: bytes) -> str:
    """Convert STL bytes into a watertight OCC solid written to a temp BREP file.

    The previous implementation embedded the STL surfaces as a 2D shell inside
    the fluid box (gmsh ``mesh.embed``), leaving fluid on both sides of the
    surface and producing physically impossible CFD results (negative drag).
    Sewing the triangulation into a closed shell and promoting it to a
    ``TopoDS_Solid`` gives us a real OCC volume that can be subtracted from
    the fluid domain via boolean cut.
    """
    try:
        from OCC.Core.BRepBuilderAPI import (  # type: ignore[import-not-found]
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCC.Core.BRepTools import breptools  # type: ignore[import-not-found]
        from OCC.Core.TopAbs import TopAbs_SHELL  # type: ignore[import-not-found]
        from OCC.Core.TopoDS import topods  # type: ignore[import-not-found]
        from OCC.Extend.DataExchange import (  # type: ignore[import-not-found]
            read_stl_file,
        )
    except ImportError as exc:
        raise_mcp_error(
            "DependencyError",
            f"pythonocc-core is required to build CFD-correct volume meshes: {exc}",
        )

    with tempfile.NamedTemporaryFile(
        suffix=".stl", delete=False, mode="wb"
    ) as stl_file:
        stl_file.write(stl_bytes)
        stl_path = stl_file.name

    try:
        triangulation = read_stl_file(stl_path)

        sewer = BRepBuilderAPI_Sewing(0.01)
        sewer.Add(triangulation)
        sewer.Perform()
        sewn = sewer.SewedShape()

        if sewn.ShapeType() == TopAbs_SHELL:
            solid = BRepBuilderAPI_MakeSolid(topods.Shell(sewn)).Solid()
        else:
            solid = sewn

        with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as brep_file:
            brep_path = brep_file.name
        breptools.Write(solid, brep_path)
        return brep_path
    finally:
        try:
            os.unlink(stl_path)
        except Exception:  # noqa: BLE001
            pass


def _generate_volume_mesh_gmsh(
    stl_bytes: bytes,
    params: GenerateVolumeMeshParams,
    brep_bytes: bytes | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Generate a CFD-ready volume mesh around the supplied geometry.

    The aircraft is imported into gmsh's OCC kernel as a watertight solid
    and subtracted from the far-field box (boolean cut) so the resulting
    fluid domain wraps the body without including any cells inside it.

    Two input paths:

    * ``brep_bytes`` (preferred): native OCC BREP exported by TiGL via
      ``exportWingBREPByUID`` / ``exportFusedBREP``. Watertight parametric
      CAD imports directly with no sewing — robust for swept wings with
      multiple sections.

    * ``stl_bytes`` (fallback): tessellated surface. ``BRepBuilderAPI_Sewing``
      stitches it into a shell. Works for simple closed bodies but can
      leak at segment seams on complex aircraft geometry. Kept so unit
      tests and synthetic inputs continue to function.

    ``tests/test_volume_mesh_cfd_correctness.py`` pins down the topology
    contract: no fluid cell may lie inside the input solid.
    """
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

    # Materialize the BREP we'll feed into gmsh's OCC importer. Prefer the
    # native CAD path if available; otherwise build one by sewing the STL.
    if brep_bytes is not None:
        with tempfile.NamedTemporaryFile(
            suffix=".brep", delete=False, mode="wb"
        ) as brep_file:
            brep_file.write(brep_bytes)
            brep_path = brep_file.name
        used_brep_source = "tigl"
    else:
        brep_path = _stl_bytes_to_solid_brep(stl_bytes)
        used_brep_source = "stl_sewn"

    stats: dict[str, Any] = {"brep_source": used_brep_source}

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volume_mesh")

        # Aircraft solid (OCC) ─────────────────────────────────────────────
        imported = gmsh.model.occ.importShapes(brep_path)
        gmsh.model.occ.synchronize()
        ac_volumes = [t for d, t in imported if d == 3]
        if not ac_volumes:
            raise_mcp_error(
                "MeshError",
                "Input geometry did not produce a closed OCC solid; cannot "
                "build CFD mesh.",
            )

        # Sizing of the far-field box derives from the OCC solid's bbox
        # rather than re-parsing the STL — works equally for both paths.
        min_x, min_y, min_z, max_x, max_y, max_z = (
            gmsh.model.occ.getBoundingBox(3, ac_volumes[0])
        )
        for tag in ac_volumes[1:]:
            bb = gmsh.model.occ.getBoundingBox(3, tag)
            min_x = min(min_x, bb[0]); min_y = min(min_y, bb[1]); min_z = min(min_z, bb[2])
            max_x = max(max_x, bb[3]); max_y = max(max_y, bb[4]); max_z = max(max_z, bb[5])
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

        # Far-field box (OCC) ─────────────────────────────────────────────
        box_tag = gmsh.model.occ.addBox(
            cx - ff, cy - ff, cz - ff, 2 * ff, 2 * ff, 2 * ff
        )
        gmsh.model.occ.synchronize()

        # Subtract the body from the box so the fluid domain wraps it
        cut_result, _ = gmsh.model.occ.cut(
            [(3, box_tag)], [(3, t) for t in ac_volumes]
        )
        gmsh.model.occ.synchronize()

        fluid_volume_tags = [t for d, t in cut_result if d == 3]

        # Classify resulting surfaces by centroid: those on the box's outer
        # planes are far-field, everything else is the body wall.
        farfield_tags: list[int] = []
        aircraft_tags: list[int] = []
        box_center = (cx, cy, cz)
        edge_tol = 0.01 * char_length
        for _, surface_tag in gmsh.model.getEntities(2):
            com = gmsh.model.occ.getCenterOfMass(2, surface_tag)
            on_box = any(
                abs(com[axis] - (box_center[axis] + sign * ff)) < edge_tol
                for axis in range(3)
                for sign in (-1, +1)
            )
            (farfield_tags if on_box else aircraft_tags).append(surface_tag)

        if farfield_tags:
            fg = gmsh.model.addPhysicalGroup(2, farfield_tags)
            gmsh.model.setPhysicalName(2, fg, "farfield")
        if aircraft_tags:
            ag = gmsh.model.addPhysicalGroup(2, aircraft_tags)
            gmsh.model.setPhysicalName(2, ag, "aircraft")
        if fluid_volume_tags:
            vg = gmsh.model.addPhysicalGroup(3, fluid_volume_tags)
            gmsh.model.setPhysicalName(3, vg, "fluid")

        stats["farfield_surfaces"] = len(farfield_tags)
        stats["aircraft_surfaces"] = len(aircraft_tags)

        # Mesh sizing — curvature-aware on the body so the LE/TE radii get
        # enough elements; bounded by the user's mesh_size_min/max.
        gmsh.option.setNumber("Mesh.MeshSizeMin", params.mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", params.mesh_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)    # Delaunay 3D

        gmsh.model.mesh.generate(3)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["node_count"] = len(node_tags)
        _, elem_tags, _ = gmsh.model.mesh.getElements(3)
        stats["element_count"] = sum(len(tags) for tags in elem_tags)

        # Write SU2 (or MSH) — only entities with physical tags are exported.
        gmsh.option.setNumber("Mesh.SaveAll", 0)
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
        for p in (stl_path, output_path, brep_path):
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

            # Prefer TiGL's native BREP (parametric CAD, watertight) and
            # fall back to STL (tessellation) only when BREP is unavailable.
            brep_bytes: bytes | None = None
            if params.component_uid is not None:
                component = config.find_component(params.component_uid)
                if component is None:
                    raise_mcp_error(
                        "NotFound", f"Component '{params.component_uid}' not found"
                    )
                brep_bytes = _export_brep_for_component(tigl_handle, component)
                stl_bytes = _export_stl_for_component(tigl_handle, component)
            else:
                brep_bytes = _export_configuration_brep(tigl_handle)
                stl_bytes = _export_configuration_stl(tigl_handle)

            if not stl_bytes or len(stl_bytes) < 100:
                raise_mcp_error(
                    "ExportError", "Failed to export geometry - empty or too small"
                )

            mesh_bytes, stats = _generate_volume_mesh_gmsh(
                stl_bytes, params, brep_bytes=brep_bytes
            )

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
