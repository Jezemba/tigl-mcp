"""An unknown component UID must name the UIDs that exist.

``set_high_level_parameters`` had the worst identical-retry rate in the system:
**25 of 50 calls (50%)** across the 2026-08 sweeps were byte-identical repeats
of the call that had just failed. The observed trace is a caller guessing names
one at a time, paying a full model step for each guess:

    component_uid='D150_Wing' -> "Component 'D150_Wing' not found"   (~91 s)
    component_uid='Wing_1'    -> "Component 'Wing_1' not found"      (~80 s)
    component_uid='Fuselage'  -> "Component 'Fuselage' not found"    (~57 s)

The real UIDs are ``Wing1`` and ``Fuselage1``. The session holds an open CPACS
configuration and ``config.all_components()`` sits one call from every one of
the nine raise sites, so the server could have answered at each of those steps
and did not. Nothing about the wording was wrong; the information was simply
withheld.
"""

from __future__ import annotations

import pytest

from tigl_mcp.errors import MCPError, raise_component_not_found


class _Component:
    def __init__(self, uid: str) -> None:
        self.uid = uid


class _Config:
    """Stands in for the loaded CPACS configuration."""

    def __init__(self, *uids: str) -> None:
        self._uids = uids

    def all_components(self):
        return [_Component(u) for u in self._uids]


F25 = _Config("Wing1", "Fuselage1", "HorizontalTail", "VerticalTail")


class _Reported:
    """MCPError carries its payload in .to_dict(), not attributes."""

    def __init__(self, error: MCPError) -> None:
        payload = error.to_dict()["error"]
        self.message = str(payload["message"])
        self.details = payload["details"]


def _raise(uid: str, config=F25, kind: str = "Component") -> _Reported:
    with pytest.raises(MCPError) as exc:
        raise_component_not_found(config, uid, kind)
    return _Reported(exc.value)


class TestTheObservedGuessesAreAnswered:
    @pytest.mark.parametrize("guess", ["D150_Wing", "Wing_1", "Fuselage"])
    def test_available_uids_are_listed(self, guess):
        msg = _raise(guess).message
        assert "Wing1" in msg and "Fuselage1" in msg

    @pytest.mark.parametrize("guess,expected", [
        ("Wing_1", "Wing1"),
        ("Fuselage", "Fuselage1"),
        ("wing1", "Wing1"),
    ])
    def test_near_miss_is_suggested(self, guess, expected):
        assert f"Did you mean '{expected}'?" in _raise(guess).message

    def test_a_wild_guess_still_gets_the_list(self):
        """No suggestion is fine; withholding the list is not."""
        msg = _raise("Nacelle_Left_Outboard").message
        assert "Wing1" in msg

    def test_requested_and_available_are_machine_readable(self):
        details = _raise("D150_Wing").details
        assert details["requested"] == "D150_Wing"
        assert "Wing1" in details["available_uids"]

    def test_uids_are_sorted_for_stable_output(self):
        msg = _raise("x", _Config("Zed", "Alpha", "Mid")).message
        assert msg.index("Alpha") < msg.index("Mid") < msg.index("Zed")


class TestKindIsReported:
    def test_wing_specific_wording(self):
        assert _raise("Wing_1", F25, "Wing").message.startswith("Wing 'Wing_1' not found")

    def test_fuselage_specific_wording(self):
        assert _raise("Fus", F25, "Fuselage").message.startswith("Fuselage 'Fus' not found")


class TestDegenerateConfigurations:
    def test_no_components_says_open_a_file(self):
        msg = _raise("Wing1", _Config()).message
        assert "no components" in msg
        assert "Open a CPACS file" in msg

    def test_reporting_never_raises_a_second_error(self):
        """A failure while reporting a failure would hide both."""

        class _Broken:
            def all_components(self):
                raise RuntimeError("tigl handle closed")

        assert "no components" in _raise("Wing1", _Broken()).message


class TestEveryCallSiteUsesIt:
    """Nine sites shared the bare wording; a missed one keeps its retry loop."""

    def test_no_bare_component_not_found_remains(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tigl_mcp" / "tools"
        offenders = [
            p.name
            for p in root.glob("*.py")
            if "' not found\"" in p.read_text() and "raise_component_not_found" not in p.read_text()
        ]
        assert not offenders, f"still raising bare not-found: {offenders}"
