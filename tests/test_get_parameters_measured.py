"""get_high_level_parameters reports the measured geometry (Avion B145, 2026-10-05).

MAS-Aviary run v2b143: get_high_level_parameters returned {"parameters": {}} for Wing1, Wing2H,
Wing3V and Fuselage1 in every attempt. It returned only numeric XML attributes of the component
element, and a real CPACS component has none. It now measures, under the keys
set_high_level_parameters takes. The D150 tests use the native runtime and are skipped without it.
"""

from __future__ import annotations

import pytest

from tests.test_planform_measurements import _session


def _get(sm, sid, uid):
    from tigl_mcp.tools.parameters import get_high_level_parameters_tool

    return get_high_level_parameters_tool(sm).handler({"session_id": sid, "component_uid": uid})


@pytest.mark.integration_real
def test_the_main_wing_is_measured_full_wing():
    sm, sid = _session()
    r = _get(sm, sid, "Wing1")
    p = r["parameters"]
    assert r["measured"] is True and "quarter-chord" in r["note"]
    assert p["area"] == pytest.approx(122.78, abs=0.05)
    assert p["aspect_ratio"] == pytest.approx(9.37, abs=0.01)
    assert p["span"] == pytest.approx(33.91, abs=0.01)
    assert p["sweep"] == pytest.approx(21.65, abs=0.1)
    assert p["taper_ratio"] == pytest.approx(0.246, abs=2e-3)


@pytest.mark.integration_real
def test_the_tails_are_measured():
    sm, sid = _session()
    ht = _get(sm, sid, "Wing2H")["parameters"]
    fin = _get(sm, sid, "Wing3V")["parameters"]
    assert ht["area"] == pytest.approx(31.1, abs=0.1)        # symmetric: both halves
    assert fin["area"] == pytest.approx(21.5, abs=0.05)      # the fin is not mirrored
    assert fin["aspect_ratio"] == pytest.approx(1.61, abs=0.01)


@pytest.mark.integration_real
def test_the_fuselage_is_measured():
    sm, sid = _session()
    p = _get(sm, sid, "Fuselage1")["parameters"]
    assert p["length"] == pytest.approx(37.68, abs=0.05)
    assert p["diameter"] == pytest.approx(4.15, abs=0.02)
    assert p["max_width"] == pytest.approx(3.96, abs=0.02)


@pytest.mark.integration_real
def test_after_a_reshape_it_reports_the_new_wing():
    from tigl_mcp.tools.parameters import set_high_level_parameters_tool

    sm, sid = _session()
    s = set_high_level_parameters_tool(sm).handler(
        {"session_id": sid, "component_uid": "Wing1", "updates": {"area": 140.0}})
    p = _get(sm, sid, "Wing1")["parameters"]
    assert p["area"] == pytest.approx(s["geometry_morph"]["after"]["reference_area"], rel=1e-6)
    assert p["area"] == pytest.approx(140.0, rel=0.02)


def test_a_stub_session_keeps_the_recorded_values():
    from types import SimpleNamespace

    from tigl_mcp.tools.parameters import _measured_parameters

    comp = SimpleNamespace(type_name="Wing", uid="W", index=1, symmetry=None, parameters={"span": 30.0})
    assert _measured_parameters(SimpleNamespace(_tigl_handle=None), comp) is None
