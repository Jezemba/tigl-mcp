"""Avion B241: generate_volume_mesh can deform ONE anchor mesh instead of re-meshing every design.

Unit tests need no TiGL. The geometry tests run only where tigl3 + pythonocc and a prepared anchor exist
(env TIGL_B241_TEST_DIR = a directory made by `prepare_anchor`, plus `cpacs/c03_ar17.xml` and `cpacs/c02_area100.xml`
morphs of the anchor); on ws8 that is b3_scratch/deform_d150_v2.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tigl_mcp.errors import MCPError
from tigl_mcp.tools import mesh_deformation as md

MESH = """NDIME= 3
NELEM= 2
10 0 1 2 3 0
10 1 2 3 4 1
NPOIN= 5
0.0 0.0 0.0 0
1.0 0.0 0.0 1
0.0 1.0 0.0 2
0.0 0.0 1.0 3
1.0 1.0 1.0 4
NMARK= 2
MARKER_TAG= farfield
MARKER_ELEMS= 1
5 1 2 4
MARKER_TAG= aircraft
MARKER_ELEMS= 2
5 0 1 2
5 0 2 3
"""


def test_reads_points_markers_and_element_count(tmp_path):
    p = tmp_path / "m.su2"
    p.write_text(MESH)
    points, markers, nelem = md.read_su2_mesh(p)
    assert points.shape == (5, 3) and nelem == 2
    assert list(markers["aircraft"]) == [0, 1, 2, 3] and list(markers["farfield"]) == [1, 2, 4]


def test_positions_file_is_whitespace_separated_global_ids(tmp_path):
    """SU2 reads it with operator>> (CSurfaceMovement::SetExternal_Deformation): commas would break it."""
    f = tmp_path / "pos.dat"
    md.write_positions(f, np.array([7, 3]), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    first = f.read_text().splitlines()[0].split()
    assert first[0] == "7" and float(first[3]) == 3.0 and "," not in f.read_text()


def test_def_config_moves_only_the_aircraft_marker():
    cfg = md.DEF_CONFIG.format(mesh="/a/anchor.su2", marker="aircraft", nonlinear=1)
    assert "DV_KIND= SURFACE_FILE" in cfg and "DV_MARKER= ( aircraft )" in cfg
    assert "MARKER_FAR= ( farfield )" in cfg and "MESH_FILENAME= /a/anchor.su2" in cfg


def test_def_log_parsing():
    ok = md.parse_def_log("Min. volume: 3.6e-06, max. volume: 896.\n... Min. volume: 3.61939e-06. Error: 9e-11.")
    assert ok == {"min_volume": pytest.approx(3.61939e-06), "negative_volume_reported": False}
    bad = md.parse_def_log("There are 12 elements with negative volume.\nMin. volume: -1e-08.")
    assert bad["negative_volume_reported"] and bad["min_volume"] < 0


def test_inverted_cells_are_refused_not_returned(tmp_path, monkeypatch):
    (tmp_path / md.MESH_FILE).write_text(MESH)
    script = tmp_path / "fake_def.sh"
    script.write_text("#!/bin/sh\ncp \"$(sed -n 's/^MESH_FILENAME= //p' $1)\" mesh_def.su2\n"
                      "echo 'There are 3 elements with negative volume.'\necho 'Min. volume: -2e-07.'\n")
    script.chmod(0o755)
    with pytest.raises(md.DeformationError, match="inverted"):
        md.deform_mesh(tmp_path, np.array([0]), np.zeros((1, 3)), su2_def=str(script), launch="")


def test_a_clean_deformation_returns_the_mesh(tmp_path):
    (tmp_path / md.MESH_FILE).write_text(MESH)
    script = tmp_path / "fake_def.sh"
    script.write_text("#!/bin/sh\ncp \"$(sed -n 's/^MESH_FILENAME= //p' $1)\" mesh_def.su2\n"
                      "echo 'Min. volume: 3.6e-06. Error: 9e-11.'\n")
    script.chmod(0o755)
    mesh, info = md.deform_mesh(tmp_path, np.array([0]), np.zeros((1, 3)), su2_def=str(script), launch="")
    assert mesh.startswith(b"NDIME=") and info["min_volume"] == pytest.approx(3.6e-06)
    assert (Path(info["workdir"]) / "surface_positions.dat").read_text().startswith("0 ")


def test_mode_is_off_unless_the_anchor_is_configured(monkeypatch):
    monkeypatch.delenv(md.ANCHOR_ENV, raising=False)
    assert md.anchor_dir_from_env() is None
    monkeypatch.setenv(md.ANCHOR_ENV, " /data/anchor ")
    assert md.anchor_dir_from_env() == "/data/anchor"


class _Handle:
    _tigl_handle = object()


def test_tool_reply_in_deformation_mode(monkeypatch):
    from tigl_mcp.tools import volume_mesh as vm

    monkeypatch.setattr(md, "deformed_mesh_for",
                        lambda tigl, anchor: (b"NDIME= 3\n", {"mesh_mode": "deformed_from_anchor",
                                                             "element_count": 2}))
    params = vm.GenerateVolumeMeshParams(session_id="s", mesh_size_min=0.33, far_field_distance=12.0,
                                         boundary_layer_enabled=False)          # False is the default: not listed
    out = vm._deformed_reply(params, _Handle(), "/anchor")
    assert out["format"] == "su2" and out["mesh_size_bytes"] == 9
    assert out["statistics"]["mesh_mode"] == "deformed_from_anchor"
    assert out["statistics"]["mesh_settings_ignored"] == ["far_field_distance", "mesh_size_min"]


def test_deformation_failure_is_a_mesh_error_not_a_silent_remesh(monkeypatch):
    from tigl_mcp.tools import volume_mesh as vm

    def boom(tigl, anchor):
        raise md.DeformationError("too far from the anchor")

    monkeypatch.setattr(md, "deformed_mesh_for", boom)
    with pytest.raises(MCPError, match="too far from the anchor"):
        vm._deformed_reply(vm.GenerateVolumeMeshParams(session_id="s"), _Handle(), "/anchor")
    with pytest.raises(MCPError, match="component_uid"):
        vm._deformed_reply(vm.GenerateVolumeMeshParams(session_id="s", component_uid="Wing1"), _Handle(), "/a")


# ---- real geometry (ws8) -------------------------------------------------------------------------------------

GEO = os.environ.get("TIGL_B241_TEST_DIR")
geometry = pytest.mark.skipif(not GEO, reason="needs TIGL_B241_TEST_DIR (prepared anchor) with tigl3 + pythonocc")


@geometry
def test_anchor_maps_onto_itself_exactly():
    handle = md.open_cpacs(Path(GEO) / md.CPACS_FILE)
    ids, xyz, stats = md.surface_positions(GEO, handle[1])
    anchor = np.load(Path(GEO) / md.MAP_FILE)["anchor_xyz"]
    assert np.abs(xyz - anchor).max() < 1e-6
    assert stats["junction_nodes"] > 0 and stats["junction_residual_max_m"] < 1e-3


@geometry
@pytest.mark.parametrize("design", ["c03_ar17", "c02_area100"])
def test_junction_nodes_land_on_the_fuselage(design):
    """AR 17 pulled the root 0.6 m off the body and area 100 sank it 0.16 m inside before the junction slide."""
    handle = md.open_cpacs(Path(GEO) / "cpacs" / f"{design}.xml")
    _, _, stats = md.surface_positions(GEO, handle[1])
    assert stats["junction_residual_max_m"] < 1e-3
    assert stats["surface_displacement_max_m"] > 1.0


@geometry
def test_junction_close_to_the_wing_root_is_found():
    """Wing3V junction at g0 0.0186 with the crossing just below it: the old fixed 0.02 step skipped g < 0 and
    refused the optimizer baseline geometry (re-swept tails, B242)."""
    design = Path(GEO) / "cpacs" / "eval_baseline_reswept.xml"
    if not design.exists():
        pytest.skip("needs cpacs/eval_baseline_reswept.xml")
    _, _, stats = md.surface_positions(GEO, md.open_cpacs(design)[1])
    assert stats["junction_residual_max_m"] < 1e-3


def test_a_folded_surface_is_detected():
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    before = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    after = before.copy()
    after[2] = [-1.0, -1.0, 0.0]                  # node 2 slides past node 0: both triangles fold over
    assert list(md.flipped_triangles(tris, before, before)) == []
    assert len(md.flipped_triangles(tris, before, after)) == 2


def test_aircraft_triangles_are_local_indices(tmp_path):
    p = tmp_path / "m.su2"
    p.write_text(MESH)
    tris = md.aircraft_triangles(p, np.array([0, 1, 2, 3]))
    assert tris.tolist() == [[0, 1, 2], [0, 2, 3]]


@geometry
def test_reswept_wing_junction_does_not_fold_the_surface():
    """The optimizer baseline (all wing sections re-swept, B242) slid the junction 0.72 m and folded 44 surface
    triangles before the slide was spread to the neighbouring wing and fuselage nodes."""
    design = Path(GEO) / "cpacs" / "eval_baseline_reswept.xml"
    if not design.exists():
        pytest.skip("needs cpacs/eval_baseline_reswept.xml")
    _, _, stats = md.surface_positions(GEO, md.open_cpacs(design)[1])
    assert stats["surface_triangles_flipped"] == 0 and stats["junction_blended_nodes"] > 0
    assert stats["junction_slide_max_m"] > 0.5
