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
    with pytest.raises(md.DesignTooFar, match="inside out"):
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
        raise md.DeformationError("SU2_DEF not found")

    monkeypatch.setattr(md, "deformed_mesh_for", boom)
    with pytest.raises(MCPError, match="TOOL failure"):
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
    assert stats["surface_triangles_flipped"] == 0 and stats["fuselage_nodes_following_junction"] > 0
    assert stats["junction_move_max_m"] > 0.5


def test_design_too_far_reads_as_an_instruction_the_agent_can_follow(monkeypatch, tmp_path):
    """Jessica 2026-11-07: never block up front (the refusal rate is measured); the agent gets a readable reason."""
    from tigl_mcp.tools import volume_mesh as vm

    (tmp_path / md.META_FILE).write_text(
        '{"baseline_planform": {"Wing1": {"area_m2": 122.78, "aspect_ratio": 9.37, "sweep_setting_deg": 27.33}}}')

    def far(tigl, anchor):
        raise md.DesignTooFar("the root of Wing1 would move 2.10 m along the fuselage and the surface would fold",
                              {"component": "Wing1", "junction_move_max_m": 2.1})

    monkeypatch.setattr(md, "deformed_mesh_for", far)
    with pytest.raises(MCPError) as info:
        vm._deformed_reply(vm.GenerateVolumeMeshParams(session_id="s"), _Handle(), str(tmp_path))
    payload = info.value.to_dict()
    text = str(payload)
    assert "MeshDeformationRefused" in text and "MESH_DEFORMATION_REFUSED" in text
    assert "too far from the study's baseline" in text and "2.10 m" in text
    assert "call generate_volume_mesh again" in text and "area 122.78" in text and "unchanged" in text


def test_tool_failure_says_it_is_not_the_design():
    msg = md.agent_message(md.DeformationError("SU2_DEF failed (exit 1)"))
    assert "not because of your design" in msg and "will not help" in msg


def test_refusal_text_starts_with_its_type_and_quotes_the_component_it_names(tmp_path):
    """Envelope 2026-11-07: the MCP client gets only the message text (not the error type), and an htail change was
    reported as 'the wing root'. The text now starts with the type and quotes the named component's baseline."""
    (tmp_path / md.META_FILE).write_text(
        '{"baseline_planform": {"Wing1": {"area_m2": 122.78, "aspect_ratio": 9.37, "sweep_setting_deg": 27.33},'
        ' "Wing2H": {"area_m2": 31.0, "aspect_ratio": 4.5, "sweep_setting_deg": 32.21}}}')
    exc = md.DesignTooFar("the root of Wing2H would move 0.60 m along the fuselage", {"component": "Wing2H"})
    msg = md.agent_message(exc, tmp_path)
    assert msg.startswith("MeshDeformationRefused: ")
    assert "baseline Wing2H is area 31.0" in msg and "Wing1" not in msg
    assert md.agent_message(md.DeformationError("x")).startswith("MeshDeformationToolFailure: ")


def test_refusal_without_planform_metadata_still_goes_out(tmp_path):
    (tmp_path / md.META_FILE).write_text('{"nelem": 1}')          # older anchor, no CPACS next to it
    msg = md.agent_message(md.DesignTooFar("the root of Wing1 would move 2 m"), tmp_path)
    assert msg.startswith("MeshDeformationRefused: ") and "call generate_volume_mesh again" in msg


def test_fold_is_attributed_to_the_nearest_junction_component():
    anchor = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]], dtype=float)
    new = anchor.copy()
    new[3:] += [0.6, 0, 0]                                           # the tail's root moves 0.6 m
    junctions = np.zeros(2, dtype=md._JUNCTION_DTYPE)
    junctions["node"], junctions["wing"] = [0, 3], [0, 1]
    comps = [{"name": "wing:Wing1"}, {"name": "wing:Wing2H"}]
    name, move = md._folding_component(np.array([[3, 4, 5]]), anchor, new, junctions, comps)
    assert name == "Wing2H" and move == pytest.approx(0.6)


def test_illinois_finds_the_bracketed_root_in_few_evaluations():
    calls = []

    def f(x):
        calls.append(x)
        return (x - 0.3137) * (1.0 + 4.0 * x)          # smooth, like a signed distance across a surface

    root = md._illinois(f, 0.0, f(0.0), 1.0, f(1.0))
    assert abs(f(root)) <= md.SLIDE_TOL_M and len(calls) < 20       # bisection to 1e-12 needs ~40


def test_anchor_from_another_junction_procedure_is_a_tool_failure(tmp_path):
    (tmp_path / md.META_FILE).write_text('{"nelem": 1}')            # no procedure stamp: older anchor
    with pytest.raises(md.DeformationError, match="re-run `prepare`") as info:
        md.surface_positions(tmp_path, None)
    assert not isinstance(info.value, md.DesignTooFar)
