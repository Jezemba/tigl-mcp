"""Avion final re-run (2026-11-18): the summaries never invent numbers, the aspect ratio is the full wing's,
and the wetted areas are real and say which convention they are in.

Unit tests use a fake TiGL handle or the no-runtime stub mode (forced, so they run on a machine with TiGL
too). The D150 tests use the native runtime and are skipped without it. No network, no live server.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tigl_mcp.cpacs import BoundingBox, ComponentDefinition, CPACSConfiguration
from tigl_mcp.errors import MCPError
from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tools import build_tools, cpacs_io

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


# ---------------------------------------------------------------- fake TiGL handle (a live geometry)


class _BrokenTigl:
    """A live TiGL handle whose every query fails."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise RuntimeError(f"{name} failed")
        return fail


class _FakeTigl:
    """Answers the wing queries like the D150 Wing1 (one side of a symmetric wing)."""

    def wingGetSpan(self, uid):  # noqa: N802
        return 33.9127

    def wingGetReferenceArea(self, index, sym):  # noqa: N802
        return 61.3908

    def wingGetMAC(self, uid):  # noqa: N802
        return (4.193, 1.0, 2.0, 3.0)

    def wingGetWettedArea(self, uid):  # noqa: N802
        return 111.579

    def wingGetSurfaceArea(self, index):  # noqa: N802
        return 130.513

    def wingGetSegmentCount(self, index):  # noqa: N802
        raise RuntimeError("no chord points in the fake")


def _live_session(native, *, symmetry="x-z-plane", params=None, kind="wing"):
    sm = SessionManager()
    comp = ComponentDefinition(uid="Wing1" if kind == "wing" else "Fuselage1", name="c", index=1,
                               type_name=kind.capitalize(), symmetry=symmetry if kind == "wing" else None,
                               parameters=dict(params or {}), bounding_box=BoundingBox.from_index(1))
    config = CPACSConfiguration(wings=[comp] if kind == "wing" else [], fuselages=[comp] if kind != "wing" else [],
                                rotors=[], engines=[])
    tixi = SimpleNamespace(_tixi_handle=None, closed=False, close=lambda: None)
    tigl = SimpleNamespace(_tigl_handle=native, closed=False, close=lambda: None)
    sid = sm.create_session(tixi, tigl, config, "<cpacs/>")
    return sm, sid, comp


def test_wing_summary_failure_is_an_error_not_a_placeholder():
    """Used to fall back to span 20 + index, area 0.8 x span, unlabelled."""
    sm, sid, _ = _live_session(_BrokenTigl(), params={"span": 40.0, "area": 130.0})
    with pytest.raises(MCPError) as err:
        _tool(build_tools(sm), "get_wing_summary").handler({"session_id": sid, "wing_uid": "Wing1"})
    assert "no values are reported" in str(err.value.error)


def test_aspect_ratio_is_the_full_wing_and_areas_are_labelled():
    sm, sid, _ = _live_session(_FakeTigl(), params={"sweep": 30.0, "dihedral": 5.0})
    s = _tool(build_tools(sm), "get_wing_summary").handler({"session_id": sid, "wing_uid": "Wing1"})
    assert s["aspect_ratio"] == pytest.approx(33.9127**2 / (2 * 61.3908))      # 9.37, not 18.73
    assert s["aspect_ratio"] == s["full_aspect_ratio"]
    assert s["reference_area"] == pytest.approx(61.3908)                          # one side, unchanged key
    assert s["full_reference_area"] == pytest.approx(122.7816)
    assert s["wetted_area"] == s["wetted_area_exposed_one_side"] == pytest.approx(111.579)
    assert s["wetted_area_exposed_full"] == pytest.approx(223.158)
    assert s["surface_area_gross_full"] == pytest.approx(261.026)
    assert "ONE side" in s["area_conventions"] and "EXPOSED" in s["area_conventions"]
    assert s["measured"] is True
    # the planform could not be measured: null and said so, never the recorded (requested) sweep
    assert s["sweep_deg"] is None and any("sweep" in w for w in s["warnings"])
    assert s["dihedral_deg"] is None


def test_unmirrored_component_full_equals_one_side():
    sm, sid, _ = _live_session(_FakeTigl(), symmetry=None)
    s = _tool(build_tools(sm), "get_wing_summary").handler({"session_id": sid, "wing_uid": "Wing1"})
    assert s["full_reference_area"] == pytest.approx(61.3908)
    assert s["wetted_area_exposed_full"] == pytest.approx(111.579)


def test_fuselage_summary_failure_is_an_error_not_a_placeholder(monkeypatch):
    """Used to fall back to length 15 + index."""
    from tigl_mcp.tools import morph

    monkeypatch.setattr(morph, "_fuselage_dims", lambda native, uid: {"length": None, "width": None,
                                                                       "height": None})
    sm, sid, _ = _live_session(object(), kind="fuselage", params={"length": 99.0})
    with pytest.raises(MCPError):
        _tool(build_tools(sm), "get_fuselage_summary").handler({"session_id": sid, "fuselage_uid": "Fuselage1"})


def test_fuselage_wetted_area_comes_from_the_measurement(monkeypatch):
    from tigl_mcp.tools import morph

    monkeypatch.setattr(morph, "_fuselage_dims", lambda native, uid: {"length": 37.68, "width": 3.96,
                                                                       "height": 4.15, "surface_area": 417.07})
    sm, sid, _ = _live_session(object(), kind="fuselage")
    f = _tool(build_tools(sm), "get_fuselage_summary").handler({"session_id": sid, "fuselage_uid": "Fuselage1"})
    assert f["wetted_area"] == pytest.approx(417.07) and f["measured"] is True
    assert "gross" in f["area_conventions"]


def test_get_parameters_measurement_failure_is_an_error():
    """parameters.py used to swallow the exception and report the recorded (requested) values."""
    sm, sid, _ = _live_session(_BrokenTigl(), params={"area": 150.0})
    with pytest.raises(MCPError) as err:
        _tool(build_tools(sm), "get_high_level_parameters").handler({"session_id": sid, "component_uid": "Wing1"})
    assert err.value.error["error"]["type"] == "MeasurementError"


def test_sampling_is_refused_on_a_live_geometry():
    sm, sid, _ = _live_session(_FakeTigl())
    with pytest.raises(MCPError) as err:
        _tool(build_tools(sm), "sample_component_surface").handler({
            "session_id": sid, "component_uid": "Wing1", "parameterization": "wing_segment_eta_xsi",
            "samples": [{"eta": 0.5, "xsi": 0.5}]})
    assert "made-up points" in str(err.value.error)


# ---------------------------------------------------------------- no TiGL runtime: recorded values, labelled


@pytest.fixture
def stub_session(monkeypatch, sample_cpacs_xml):
    monkeypatch.setattr(cpacs_io, "_load_real_bindings", lambda: (None, None))
    sm = SessionManager()
    tools = build_tools(sm)
    out = _tool(tools, "open_cpacs").handler({"source_type": "xml_string", "source": sample_cpacs_xml})
    assert out["geometry_runtime"].startswith("none")
    return tools, out["session_id"]


def test_stub_summaries_are_labelled_and_invent_nothing(stub_session):
    tools, sid = stub_session
    w = _tool(tools, "get_wing_summary").handler({"session_id": sid, "wing_uid": "W1"})
    assert w["measured"] is False and "NOT measurements" in w["note"]
    assert w["span"] == 30.0 and w["reference_area"] == 80.0          # recorded in the fixture
    assert w["mac_length"] is None and w["mac_quarter_chord"] is None  # not recorded: null
    f = _tool(tools, "get_fuselage_summary").handler({"session_id": sid, "fuselage_uid": "F1"})
    assert f["measured"] is False and f["length"] == 25.0 and f["wetted_area"] is None
    p = _tool(tools, "get_high_level_parameters").handler({"session_id": sid, "component_uid": "W1"})
    assert p["measured"] is False


def test_stub_wing_without_recorded_span_reports_null_not_20(monkeypatch):
    monkeypatch.setattr(cpacs_io, "_load_real_bindings", lambda: (None, None))
    sm = SessionManager()
    tools = build_tools(sm)
    xml = '<cpacs><vehicles><aircraft><model><wings><wing uID="W"/></wings></model></aircraft></vehicles></cpacs>'
    sid = _tool(tools, "open_cpacs").handler({"source_type": "xml_string", "source": xml})["session_id"]
    w = _tool(tools, "get_wing_summary").handler({"session_id": sid, "wing_uid": "W"})
    assert w["span"] is None and w["reference_area"] is None and w["aspect_ratio"] is None


def test_stub_boxes_and_counts_are_null_not_made_up(stub_session):
    tools, sid = stub_session
    s = _tool(tools, "get_configuration_summary").handler({"session_id": sid})
    assert s["bounding_box"] is None and "not measured" in s["bounding_box_note"]
    m = _tool(tools, "get_component_metadata").handler({"session_id": sid, "component_uid": "W1"})
    assert m["bounding_box"] is None and m["wing_data"]["num_segments"] is None
    pts = _tool(tools, "sample_component_surface").handler({
        "session_id": sid, "component_uid": "W1", "parameterization": "wing_segment_eta_xsi",
        "samples": [{"eta": 0.5, "xsi": 0.5}]})
    assert pts["synthetic"] is True and "SYNTHETIC" in pts["note"]


# ---------------------------------------------------------------- D150 through the real runtime


def _d150():
    pytest.importorskip("tigl3")
    if not CPACS.exists():
        pytest.skip("D150 fixture not available")
    sm = SessionManager()
    tools = build_tools(sm)
    sid = _tool(tools, "open_cpacs").handler({"source_type": "path", "source": str(CPACS)})["session_id"]
    if sm.get(sid)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")
    return tools, sid


@pytest.mark.integration_real
def test_d150_wing_summary_full_ar_and_wetted_areas():
    tools, sid = _d150()
    s = _tool(tools, "get_wing_summary").handler({"session_id": sid, "wing_uid": "Wing1"})
    assert s["aspect_ratio"] == pytest.approx(9.37, abs=0.01)
    assert s["wetted_area"] == pytest.approx(111.58, abs=0.05)              # one side, exposed
    assert s["surface_area_gross_one_side"] == pytest.approx(130.51, abs=0.05)
    assert s["wetted_area_exposed_full"] == pytest.approx(223.16, abs=0.1)


@pytest.mark.integration_real
def test_d150_fuselage_wetted_area_is_measured():
    """48 of 48 l20_final get_fuselage_summary calls returned wetted_area null."""
    tools, sid = _d150()
    f = _tool(tools, "get_fuselage_summary").handler({"session_id": sid, "fuselage_uid": "Fuselage1"})
    assert f["wetted_area"] == pytest.approx(417.07, abs=0.1)


@pytest.mark.integration_real
def test_d150_whole_aircraft_swet_sref_is_about_six():
    """Exposed wings + tails (both sides where mirrored) + fuselage over the full reference area."""
    tools, sid = _d150()
    ws = _tool(tools, "get_wing_summary")
    total = sum(ws.handler({"session_id": sid, "wing_uid": u})["wetted_area_exposed_full"]
                for u in ("Wing1", "Wing2H", "Wing3V"))
    total += _tool(tools, "get_fuselage_summary").handler({"session_id": sid,
                                                            "fuselage_uid": "Fuselage1"})["wetted_area"]
    sref = ws.handler({"session_id": sid, "wing_uid": "Wing1"})["full_reference_area"]
    assert total / sref == pytest.approx(5.99, abs=0.05)


@pytest.mark.integration_real
def test_d150_configuration_box_is_measured():
    tools, sid = _d150()
    s = _tool(tools, "get_configuration_summary").handler({"session_id": sid})
    box = s["bounding_box"]
    assert box is not None and "measured" in s["bounding_box_note"]
    assert box["ymax"] - box["ymin"] == pytest.approx(33.91, abs=0.1)        # mirrored wing: tip to tip
    assert box["xmax"] - box["xmin"] >= 37.6                                 # at least the fuselage length
    m = _tool(tools, "get_component_metadata").handler({"session_id": sid, "component_uid": "Wing1"})
    assert m["wing_data"]["num_segments"] >= 1
