"""Planform measurements: taper ratio and sweep from TiGL chord points (Avion B133), the measured
fuselage summary, and the ignored-key warning on set_high_level_parameters (Avion B105).

The pure-geometry tests need no TiGL. The D150 tests use the native runtime and are skipped
without it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tigl_mcp.tools.planform import planform_from_sections

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"


# ---- pure geometry ----------------------------------------------------------------------------

def _trapezoid(root_chord, tip_chord, semi_span, le_sweep_deg, x0=10.0):
    dx = semi_span * math.tan(math.radians(le_sweep_deg))
    root = ((x0, 0.0, 0.0), (x0 + root_chord, 0.0, 0.0))
    tip = ((x0 + dx, semi_span, 0.0), (x0 + dx + tip_chord, semi_span, 0.0))
    return [root, tip]


def test_a_straight_tapered_wing_gives_its_taper_and_sweeps():
    m = planform_from_sections(_trapezoid(6.0, 1.5, 16.0, 30.0))
    assert m["taper_ratio"] == pytest.approx(0.25)
    assert m["sweep_le_deg"] == pytest.approx(30.0)
    # c/4 line: dx_c4 = dx_le + 0.25 * (tip - root) chord
    expected = math.degrees(math.atan((16.0 * math.tan(math.radians(30.0)) + 0.25 * (1.5 - 6.0)) / 16.0))
    assert m["sweep_quarter_chord_deg"] == pytest.approx(expected)
    assert m["span_axis"] == "y"


def test_an_unswept_rectangle_has_zero_sweep_and_unit_taper():
    m = planform_from_sections(_trapezoid(4.0, 4.0, 10.0, 0.0))
    assert m["taper_ratio"] == pytest.approx(1.0)
    assert m["sweep_le_deg"] == pytest.approx(0.0) and m["sweep_quarter_chord_deg"] == pytest.approx(0.0)


def test_a_vertical_fin_is_measured_along_z():
    root = ((30.0, 0.0, 0.0), (35.0, 0.0, 0.0))
    tip = ((33.0, 0.0, 5.0), (35.0, 0.0, 5.0))
    m = planform_from_sections([root, tip])
    assert m["span_axis"] == "z"
    assert m["sweep_le_deg"] == pytest.approx(math.degrees(math.atan(3.0 / 5.0)))
    assert m["taper_ratio"] == pytest.approx(0.4)


def test_a_zero_span_segment_is_skipped_in_the_panels():
    root = ((12.0, 0.0, 0.0), (18.0, 0.0, 0.0))
    same = ((12.0, 0.0, 0.0), (18.0, 0.0, 0.0))
    tip = ((20.0, 16.0, 0.0), (21.5, 16.0, 0.0))
    m = planform_from_sections([root, same, tip])
    assert len(m["panels"]) == 1


def test_fewer_than_two_sections_is_refused():
    with pytest.raises(ValueError):
        planform_from_sections([((0, 0, 0), (1, 0, 0))])


# ---- D150 through the real tools ---------------------------------------------------------------

def _session():
    pytest.importorskip("tigl3")
    from tigl_mcp.session_manager import SessionManager
    from tigl_mcp.tools.cpacs_io import open_cpacs_tool

    if not CPACS.exists():
        pytest.skip("D150 fixture not available")
    sm = SessionManager()
    sid = open_cpacs_tool(sm).handler({"source_type": "path", "source": str(CPACS)})["session_id"]
    if sm.get(sid)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")
    return sm, sid


@pytest.mark.integration_real
def test_d150_wing_summary_reports_measured_planform():
    from tigl_mcp.tools.metrics import get_wing_summary_tool

    sm, sid = _session()
    s = get_wing_summary_tool(sm).handler({"session_id": sid, "wing_uid": "Wing1"})
    assert s["taper_ratio"] == pytest.approx(0.246, abs=2e-3)
    assert s["sweep_le_deg"] == pytest.approx(24.9, abs=0.1)
    assert s["sweep_quarter_chord_deg"] == pytest.approx(21.65, abs=0.1)
    assert s["sweep_deg"] == s["sweep_quarter_chord_deg"]          # no longer null
    assert s["full_reference_area"] == pytest.approx(2 * s["reference_area"])
    assert s["full_aspect_ratio"] == pytest.approx(9.37, abs=0.01)


@pytest.mark.integration_real
def test_d150_tails_are_measured_too():
    from tigl_mcp.tools.metrics import get_wing_summary_tool

    sm, sid = _session()
    ht = get_wing_summary_tool(sm).handler({"session_id": sid, "wing_uid": "Wing2H"})
    fin = get_wing_summary_tool(sm).handler({"session_id": sid, "wing_uid": "Wing3V"})
    assert ht["taper_ratio"] == pytest.approx(0.33, abs=5e-3)
    assert ht["sweep_le_deg"] == pytest.approx(32.3, abs=0.2)
    assert fin["sweep_le_deg"] is not None and fin["taper_ratio"] == pytest.approx(0.349, abs=5e-3)


@pytest.mark.integration_real
def test_a_morph_reports_the_achieved_sweep_and_taper():
    from tigl_mcp.tools.morph import morph_wing_tool

    sm, sid = _session()
    r = morph_wing_tool(sm).handler({"session_id": sid, "wing_uid": "Wing1", "target_area_m2": 140.0})
    assert r["before"]["taper_ratio"] == pytest.approx(0.246, abs=2e-3)
    assert r["after"]["sweep_deg"] is not None and r["after"]["taper_ratio"] is not None
    # an area-only morph scales chords uniformly: taper unchanged
    assert r["after"]["taper_ratio"] == pytest.approx(r["before"]["taper_ratio"], abs=2e-3)


@pytest.mark.integration_real
def test_d150_fuselage_summary_is_measured_not_stubbed():
    from tigl_mcp.tools.metrics import get_fuselage_summary_tool

    sm, sid = _session()
    f = get_fuselage_summary_tool(sm).handler({"session_id": sid, "fuselage_uid": "Fuselage1"})
    assert f["measured"] is True
    assert f["length"] == pytest.approx(37.68, abs=0.05)
    assert f["max_width"] == pytest.approx(3.96, abs=0.02) and f["max_height"] == pytest.approx(4.15, abs=0.02)
    assert f["max_diameter"] == pytest.approx(4.15, abs=0.02)


@pytest.mark.integration_real
def test_ignored_keys_are_reported():
    from tigl_mcp.tools.parameters import set_high_level_parameters_tool

    sm, sid = _session()
    tool = set_high_level_parameters_tool(sm)
    r = tool.handler({"session_id": sid, "component_uid": "Wing1",
                      "updates": {"area": 125.0, "taper_ratio": 0.3, "dihedral": 4.0}})
    assert any("Ignored" in w and "taper_ratio" in w and "dihedral" in w for w in r["warnings"])
    assert "geometry_morph" in r                                    # the real key still morphed
    r2 = tool.handler({"session_id": sid, "component_uid": "Wing1", "updates": {"span": 40.0}})
    assert any("Ignored" in w and "span" in w for w in r2["warnings"])   # span alone moves nothing
