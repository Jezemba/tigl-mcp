"""Avion V5 follow-up (2026-11-18): applying the measured geometry is a reported no-op that leaves the CPACS untouched.

The fuselage morph had no tolerance: re-applying the measured D150 fuselage (37.6446 m long, 4.1437 m diameter)
rescaled it by ~1e-6, rewrote the CPACS and changed the design fingerprint (results then read as stale after a
no-op). Wings and tails already held targets within 1e-4 but reported a rebuild. Needs the native TiGL runtime.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tools.cpacs_io import export_cpacs_tool, open_cpacs_tool
from tigl_mcp.tools.morph import NO_CHANGE_NOTE
from tigl_mcp.tools.parameters import get_high_level_parameters_tool, set_high_level_parameters_tool

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"
pytestmark = pytest.mark.integration_real


def _session():
    pytest.importorskip("tigl3")
    if not CPACS.exists():
        pytest.skip("D150 CPACS fixture not available")
    sm = SessionManager()
    sid = open_cpacs_tool(sm).handler({"source_type": "path", "source": str(CPACS)})["session_id"]
    if sm.get(sid)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")
    return sm, sid


def _measured(sm, sid, uid):
    return get_high_level_parameters_tool(sm).handler({"session_id": sid, "component_uid": uid})["parameters"]


def _set(sm, sid, uid, **updates):
    return set_high_level_parameters_tool(sm).handler({"session_id": sid, "component_uid": uid, "updates": updates})


def _export(sm, sid, path):
    export_cpacs_tool(sm).handler({"session_id": sid, "output_path": str(path)})
    return Path(path).read_bytes()


def test_applying_the_measured_design_leaves_the_cpacs_byte_identical(tmp_path):
    sm, sid = _session()
    untouched = _export(sm, sid, tmp_path / "untouched.xml")
    f = _measured(sm, sid, "Fuselage1")
    assert f["length"] == pytest.approx(37.6446, abs=1e-3) and f["diameter"] == pytest.approx(4.1437, abs=1e-3)
    gm = _set(sm, sid, "Fuselage1", length=f["length"], diameter=f["diameter"])["geometry_morph"]
    assert gm["no_op"] is True and gm["rebuilt"] is False and NO_CHANGE_NOTE in gm["note"]
    # the rounded values a caller quotes (37.6446 / 4.1437) are within tolerance too
    gm = _set(sm, sid, "Fuselage1", length=37.6446, diameter=4.1437)["geometry_morph"]
    assert gm["no_op"] is True and NO_CHANGE_NOTE in gm["note"]
    for uid in ("Wing1", "Wing2H", "Wing3V"):           # main wing and both tails
        w = _measured(sm, sid, uid)
        gm = _set(sm, sid, uid, area=w["area"], aspect_ratio=w["aspect_ratio"], sweep=w["sweep"])["geometry_morph"]
        assert gm["no_op"] is True and gm["rebuilt"] is False and NO_CHANGE_NOTE in gm["note"], uid
    assert _export(sm, sid, tmp_path / "reapplied.xml") == untouched
    assert _measured(sm, sid, "Fuselage1") == f


def test_a_real_change_still_morphs_and_one_axis_can_be_held(tmp_path):
    sm, sid = _session()
    untouched = _export(sm, sid, tmp_path / "untouched.xml")
    f = _measured(sm, sid, "Fuselage1")
    gm = _set(sm, sid, "Fuselage1", length=38.0, diameter=f["diameter"])["geometry_morph"]
    assert gm["rebuilt"] is True and gm["no_op"] is False
    assert gm["tolerance"]["held_within_tolerance"] == ["diameter"] and NO_CHANGE_NOTE in gm["note"]
    assert gm["after"]["length"] == pytest.approx(38.0, abs=1e-3)
    assert max(gm["after"]["width"], gm["after"]["height"]) == pytest.approx(f["diameter"], abs=1e-9)
    assert _export(sm, sid, tmp_path / "morphed.xml") != untouched
