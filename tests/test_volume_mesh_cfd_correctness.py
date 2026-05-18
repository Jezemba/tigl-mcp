"""Regression tests for generate_volume_mesh CFD correctness.

These tests guard against the Phase F/G regression where the wing surface
was embedded as a 2D shell inside the fluid volume rather than carved out
of it, producing meshes with fluid cells *inside* the solid body. Such
meshes drive SU2 to compute nonphysical CL/CD (e.g. negative drag) and
silently corrupt the entire MDO pipeline downstream.

The contract these tests pin down:

* A volume mesh produced from a watertight solid input must have zero
  fluid nodes strictly inside that solid.
* The mesh must carry the expected named markers (``aircraft`` for the
  body, ``farfield`` for the outer box).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")

from tigl_mcp.tools.volume_mesh import (  # noqa: E402  (after importorskip)
    GenerateVolumeMeshParams,
    _generate_volume_mesh_gmsh,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _icosphere_stl(radius: float = 1.0, subdiv: int = 2) -> bytes:
    """Build an ASCII STL of a watertight icosphere.

    Used as a stand-in for a wing/fuselage solid in tests so the mesher
    has a known closed surface to operate on without depending on TiGL.
    """
    import math

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]):
        return tuple((a[i] + b[i]) / 2.0 for i in range(3))

    for _ in range(subdiv):
        new_faces = []
        cache: dict[tuple[int, int], int] = {}

        def edge(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key in cache:
                return cache[key]
            m = _midpoint(verts[i], verts[j])
            verts.append(m)
            cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab = edge(a, b)
            bc = edge(b, c)
            ca = edge(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces

    # Normalise to the sphere and scale to requested radius
    normed = []
    for x, y, z in verts:
        n = math.sqrt(x * x + y * y + z * z)
        normed.append((radius * x / n, radius * y / n, radius * z / n))

    lines = ["solid icosphere"]
    for a, b, c in faces:
        v0, v1, v2 = normed[a], normed[b], normed[c]
        # Outward normal via cross product
        e1 = tuple(v1[i] - v0[i] for i in range(3))
        e2 = tuple(v2[i] - v0[i] for i in range(3))
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        lines.append(f"facet normal {nx/ln} {ny/ln} {nz/ln}")
        lines.append("  outer loop")
        for v in (v0, v1, v2):
            lines.append(f"    vertex {v[0]} {v[1]} {v[2]}")
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append("endsolid icosphere")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_su2_mesh(mesh_bytes: bytes) -> dict:
    """Parse a SU2 mesh string into points and per-marker node ids."""
    text = mesh_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    n = len(lines)

    points: list[tuple[float, float, float]] = []
    marker_nodes: dict[str, set[int]] = {}

    i = 0
    while i < n:
        line = lines[i].strip()
        if line.startswith("NPOIN"):
            npoint = int(line.split("=")[1])
            for j in range(npoint):
                parts = lines[i + 1 + j].split()
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            i += 1 + npoint
            continue
        if line.startswith("MARKER_TAG"):
            tag = line.split("=")[1].strip()
            i += 1
            nelem = int(lines[i].split("=")[1])
            ids: set[int] = set()
            for k in range(nelem):
                row = lines[i + 1 + k].split()
                etype = int(row[0])
                if etype == 5:    # triangle
                    ids.update(int(x) for x in row[1:4])
                elif etype == 9:  # quad
                    ids.update(int(x) for x in row[1:5])
            marker_nodes[tag] = ids
            i += 1 + nelem
            continue
        i += 1

    return {"points": np.array(points), "markers": marker_nodes}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_volume_mesh_excludes_fluid_inside_solid_body(tmp_path: Path) -> None:
    """No fluid mesh node may lie strictly inside the input solid body.

    This guards Phase F/G: prior implementation embedded the surface as
    a 2D shell, leaving fluid on both sides. SU2 then integrated pressure
    on both faces and produced negative drag.
    """
    radius = 1.0
    stl_bytes = _icosphere_stl(radius=radius, subdiv=2)

    params = GenerateVolumeMeshParams(
        session_id="t",
        far_field_distance=4.0,
        mesh_size_min=0.1,
        mesh_size_max=0.8,
        surface_mesh_size=0.2,
        boundary_layer_enabled=False,
        boundary_layer_thickness=0.01,
        output_format="su2",
    )
    mesh_bytes, stats = _generate_volume_mesh_gmsh(stl_bytes, params)

    parsed = _parse_su2_mesh(mesh_bytes)
    points = parsed["points"]
    assert points.size, "Mesh produced no volume nodes"

    # Distance from origin: anything strictly inside the unit sphere is
    # a fluid node that physically should not exist.
    r = np.linalg.norm(points, axis=1)
    interior_tolerance = 0.95 * radius
    interior_count = int(np.sum(r < interior_tolerance))

    assert interior_count == 0, (
        f"Mesh has {interior_count} fluid nodes inside the solid body "
        f"(r < {interior_tolerance}). Solid was not properly subtracted "
        f"from the fluid domain. min(r)={r.min():.3f}"
    )


def test_volume_mesh_carries_expected_markers() -> None:
    """The SU2 mesh must expose ``aircraft`` and ``farfield`` markers."""
    stl_bytes = _icosphere_stl(radius=1.0, subdiv=2)
    params = GenerateVolumeMeshParams(
        session_id="t",
        far_field_distance=4.0,
        mesh_size_min=0.1,
        mesh_size_max=0.8,
        output_format="su2",
    )
    mesh_bytes, stats = _generate_volume_mesh_gmsh(stl_bytes, params)

    parsed = _parse_su2_mesh(mesh_bytes)
    markers = parsed["markers"]

    assert "aircraft" in markers, f"Missing 'aircraft' marker: {sorted(markers)}"
    assert "farfield" in markers, f"Missing 'farfield' marker: {sorted(markers)}"
    assert len(markers["aircraft"]) > 0, "aircraft marker has no nodes"
    assert len(markers["farfield"]) > 0, "farfield marker has no nodes"

    # Box has 6 outer faces. The classifier must put all of them in
    # ``farfield`` — a regression in Phase G.1 silently misclassified
    # box faces as ``aircraft`` because the per-axis centre coordinate
    # was wrong, which propagated into the SU2 force computation.
    assert stats["farfield_surfaces"] == 6, (
        f"Expected 6 farfield surfaces (the box faces), got "
        f"{stats['farfield_surfaces']}. Phase G.2 classifier regression."
    )


def test_volume_mesh_via_brep_path_is_watertight() -> None:
    """Phase G.2: the BREP path must produce a leak-free mesh.

    The Phase G.1 STL-sewing path leaked on swept multi-segment wings.
    The fix routes TiGL's native BREP through gmsh.occ.importShapes,
    skipping the sewer entirely. Smoke-test that contract here with a
    synthetic OCC sphere BREP.
    """
    try:
        from OCC.Core.BRepPrimAPI import (  # type: ignore[import-not-found]
            BRepPrimAPI_MakeSphere,
        )
        from OCC.Core.BRepTools import breptools  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("pythonocc-core not available")

    import tempfile

    radius = 1.0
    sphere = BRepPrimAPI_MakeSphere(radius).Shape()
    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        brep_path = f.name
    breptools.Write(sphere, brep_path)
    with open(brep_path, "rb") as fh:
        brep_bytes = fh.read()

    params = GenerateVolumeMeshParams(
        session_id="t",
        far_field_distance=4.0,
        mesh_size_min=0.05,
        mesh_size_max=0.8,
        output_format="su2",
    )
    # The function signature requires stl_bytes; pass a small valid one
    # purely to satisfy the path even though brep_bytes drives meshing.
    stl_bytes = _icosphere_stl(radius=1.0, subdiv=1)
    mesh_bytes, stats = _generate_volume_mesh_gmsh(
        stl_bytes, params, brep_bytes=brep_bytes
    )

    assert stats["brep_source"] == "tigl", (
        "BREP-bytes path should have set brep_source=tigl in stats"
    )

    parsed = _parse_su2_mesh(mesh_bytes)
    r = np.linalg.norm(parsed["points"], axis=1)
    interior = int(np.sum(r < 0.95 * radius))
    assert interior == 0, (
        f"BREP path leak: {interior} fluid nodes inside unit sphere "
        f"(min r={r.min():.3f})"
    )
    assert stats["farfield_surfaces"] == 6, (
        f"BREP path: expected 6 box faces, got {stats['farfield_surfaces']}"
    )
