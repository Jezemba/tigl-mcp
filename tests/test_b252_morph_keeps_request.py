"""Avion B252: a wing morph gives the wing that was asked for, every time, on the real D150.

Before the fix, sending {area 130, aspect_ratio 10.5} three times gave three wings (quarter-chord sweep
21.65 -> 19.49 -> 14.67 -> 10.82 deg), and sending the wing's own values reshaped it (area 122.83 -> 125.86).
These tests fail on that code. Needs the native TiGL runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tigl3")

from tigl_mcp.session_manager import SessionManager  # noqa: E402
from tigl_mcp.tools.cpacs_io import open_cpacs_tool  # noqa: E402
from tigl_mcp.tools.parameters import get_high_level_parameters_tool, set_high_level_parameters_tool  # noqa: E402

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"

pytestmark = pytest.mark.integration_real


class Wing:
    """A real session on the D150: morph one component, read back what it measured."""

    def __init__(self, uid: str = "Wing1") -> None:
        if not CPACS.exists():
            pytest.skip("D150 CPACS fixture not available")
        self.sm = SessionManager()
        self.sid = open_cpacs_tool(self.sm).handler({"source_type": "path", "source": str(CPACS)})["session_id"]
        if self.sm.get(self.sid)[1]._tigl_handle is None:
            pytest.skip("native TiGL runtime not available")
        self.uid = uid
        self.tool = set_high_level_parameters_tool(self.sm)
        self.base = get_high_level_parameters_tool(self.sm).handler(
            {"session_id": self.sid, "component_uid": uid}
        )["parameters"]

    def morph(self, **updates: float) -> dict:
        result = self.tool.handler({"session_id": self.sid, "component_uid": self.uid, "updates": updates})
        assert result["warnings"] == []
        return result["geometry_morph"]


def _is(after: dict, area: float, ar: float, sweep: float) -> None:
    assert after["reference_area"] == pytest.approx(area, rel=2e-4)
    assert after["aspect_ratio"] == pytest.approx(ar, rel=2e-4)
    assert after["sweep_deg"] == pytest.approx(sweep, abs=0.02)


def test_the_same_request_three_times_gives_the_same_wing() -> None:
    wing = Wing()
    for _ in range(3):
        morph = wing.morph(area=130.0, aspect_ratio=10.5)
        assert morph["targets_met"] is True
        _is(morph["after"], 130.0, 10.5, wing.base["sweep"])


def test_the_wings_own_values_leave_it_unchanged() -> None:
    wing = Wing()
    area, ar, sweep = wing.base["area"], wing.base["aspect_ratio"], wing.base["sweep"]
    morph = wing.morph(area=area, aspect_ratio=ar, sweep=sweep)
    _is(morph["after"], area, ar, sweep)
    assert morph["after"]["span"] == pytest.approx(morph["before"]["span"], rel=1e-4)
    assert morph["after"]["taper_ratio"] == pytest.approx(morph["before"]["taper_ratio"], rel=1e-6)


def test_one_key_changes_only_that_key() -> None:
    wing = Wing()
    ar, sweep = wing.base["aspect_ratio"], wing.base["sweep"]
    _is(wing.morph(area=130.0)["after"], 130.0, ar, sweep)
    _is(wing.morph(aspect_ratio=10.5)["after"], 130.0, 10.5, sweep)
    _is(wing.morph(sweep=25.0)["after"], 130.0, 10.5, 25.0)


def test_all_three_keys_are_reached_together() -> None:
    morph = Wing().morph(area=118.0, aspect_ratio=11.0, sweep=27.0)
    assert morph["targets_met"] is True
    _is(morph["after"], 118.0, 11.0, 27.0)


def test_a_sweep_change_keeps_the_kink_and_the_planform() -> None:
    wing = Wing()
    native = wing.sm.get(wing.sid)[1]._tigl_handle
    before = [native.wingGetChordPoint(1, seg, 1.0, 0.0) for seg in (1, 2, 3)]
    root = native.wingGetChordPoint(1, 1, 0.0, 0.0)
    morph = wing.morph(sweep=28.0)
    native = wing.sm.get(wing.sid)[1]._tigl_handle
    after = [native.wingGetChordPoint(1, seg, 1.0, 0.0) for seg in (1, 2, 3)]
    _is(morph["after"], wing.base["area"], wing.base["aspect_ratio"], 28.0)
    assert morph["after"]["taper_ratio"] == pytest.approx(morph["before"]["taper_ratio"], rel=1e-6)
    # A shear: each section moves aft in proportion to how far out it is, so the panels stay straight and distinct.
    moved = [(a[0] - b[0]) / (b[1] - root[1]) for a, b in zip(after, before)]
    assert moved[0] == pytest.approx(moved[1], rel=0.02) and moved[1] == pytest.approx(moved[2], rel=0.02)
    assert moved[0] > 0


@pytest.mark.parametrize("uid", ["Wing2H", "Wing3V"])
def test_the_tails_hold_their_sweep_too(uid: str) -> None:
    wing = Wing(uid)
    first = wing.morph(area=30.0 if uid == "Wing2H" else 22.0)
    sweep = first["before"]["sweep_deg"]
    ar = first["before"]["aspect_ratio"]
    assert first["targets_met"] is True
    assert first["after"]["sweep_deg"] == pytest.approx(sweep, abs=0.02)
    second = wing.morph(aspect_ratio=ar * 1.1)
    assert second["targets_met"] is True
    assert second["after"]["sweep_deg"] == pytest.approx(sweep, abs=0.02)
    assert second["after"]["aspect_ratio"] == pytest.approx(ar * 1.1, rel=2e-4)


def test_a_relative_update_is_relative_to_the_measured_geometry() -> None:
    wing = Wing()
    morph = wing.morph(area="+5")  # type: ignore[arg-type]
    assert morph["after"]["reference_area"] == pytest.approx(wing.base["area"] + 5.0, rel=2e-4)


def test_a_fuselage_call_does_not_replay_an_earlier_request() -> None:
    wing = Wing("Fuselage1")
    first = wing.morph(length=40.0)
    assert first["after"]["length"] == pytest.approx(40.0, rel=2e-3)
    second = wing.morph(diameter=4.3)
    assert second["after"]["length"] == pytest.approx(40.0, rel=2e-3)
