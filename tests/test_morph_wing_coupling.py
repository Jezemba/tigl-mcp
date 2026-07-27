"""Regression test for the geometry-coupling fix (morph_wing).

Pins down the contract that ``morph_wing`` actually DEFORMS the wing so the
aero<->geometry loop is closed — the opposite of the old ``set_high_level_parameters``
no-op, where every run meshed/solved the same baseline wing.

The test asserts, on the real D150 CPACS via native TiGL, that after a morph:
  * ``get_wing_summary`` reports the NEW span, reference area, and aspect ratio
    (i.e. the read-back is not the stale cached baseline);
  * the achieved area and AR match the requested targets within tolerance;
  * ``exportWingBREPByUID`` produces DIFFERENT bytes (so ``generate_volume_mesh``
    and hence SU2 will see a different shape).

Requires the native tigl3 runtime (skipped otherwise), so it is marked
``integration_real``.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("tigl3")

from tigl_mcp.session_manager import SessionManager  # noqa: E402
from tigl_mcp.tools.cpacs_io import open_cpacs_tool  # noqa: E402
from tigl_mcp.tools.metrics import get_wing_summary_tool  # noqa: E402
from tigl_mcp.tools.morph import morph_fuselage_tool, morph_wing_tool  # noqa: E402

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"
WING = "Wing1"
FUSELAGE = "Fuselage1"

pytestmark = pytest.mark.integration_real


def _brep_bytes(session_manager: SessionManager, session_id: str) -> bytes:
    tigl = session_manager.get(session_id)[1]._tigl_handle
    path = tempfile.mktemp(suffix=".brep")
    tigl.exportWingBREPByUID(WING, path)
    return Path(path).read_bytes()


def test_morph_wing_closes_geometry_loop() -> None:
    if not CPACS.exists():
        pytest.skip("D150 CPACS fixture not available")

    sm = SessionManager()
    open_res = open_cpacs_tool(sm).handler(
        {"source_type": "path", "source": str(CPACS)}
    )
    session_id = open_res["session_id"]

    # If native TiGL isn't actually backing the session, there's nothing to
    # deform — skip rather than assert on stub numbers.
    if sm.get(session_id)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")

    summary = get_wing_summary_tool(sm)
    morph = morph_wing_tool(sm)

    ws_before = summary.handler({"session_id": session_id, "wing_uid": WING})
    brep_before = _brep_bytes(sm, session_id)

    # Aviary's F25 design variables, passed DIRECTLY (the tool is symmetry-aware,
    # so target_area_m2 / target_aspect_ratio are the full-planform convention —
    # no ratios or unit conversions needed).
    target_area = 130.1
    target_ar = 15.6
    result = morph.handler(
        {
            "session_id": session_id,
            "wing_uid": WING,
            "target_area_m2": target_area,
            "target_aspect_ratio": target_ar,
        }
    )
    assert result["rebuilt"] is True
    before, after = result["before"], result["after"]

    # 1. Baseline reads as an A320-class wing (full planform), not the half-area
    #    artifact — confirms the symmetry factor.
    assert before["aspect_ratio"] == pytest.approx(9.4, rel=0.1)

    # 2. The morph hits aviary's targets directly.
    assert after["reference_area"] == pytest.approx(target_area, rel=0.05)
    assert after["aspect_ratio"] == pytest.approx(target_ar, rel=0.05)

    # 3. get_wing_summary reflects the new geometry (span moved; not a no-op).
    ws_after = summary.handler({"session_id": session_id, "wing_uid": WING})
    assert abs(ws_after["span"] - ws_before["span"]) > 1.0

    # 4. The exported BREP (the mesher's input) actually changed → the mesh and
    #    therefore SU2's CL/CD will move off the fixed baseline.
    brep_after = _brep_bytes(sm, session_id)
    assert hashlib.md5(brep_before).digest() != hashlib.md5(brep_after).digest()


def test_morph_fuselage_closes_geometry_loop() -> None:
    if not CPACS.exists():
        pytest.skip("D150 CPACS fixture not available")

    sm = SessionManager()
    session_id = open_cpacs_tool(sm).handler(
        {"source_type": "path", "source": str(CPACS)}
    )["session_id"]
    if sm.get(session_id)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")

    morph = morph_fuselage_tool(sm)
    result = morph.handler(
        {
            "session_id": session_id,
            "fuselage_uid": FUSELAGE,
            "target_length_m": 45.0,
            "target_diameter_m": 5.0,
        }
    )
    if result.get("before", {}).get("length") is None:
        pytest.skip("OpenCASCADE not available for fuselage bbox measurement")

    assert result["rebuilt"] is True
    before, after = result["before"], result["after"]
    # Length moved toward the target and diameter grew.
    assert after["length"] == pytest.approx(45.0, rel=0.1)
    assert abs(after["length"] - before["length"]) > 1.0
    assert max(after["width"], after["height"]) > max(before["width"], before["height"])


def test_set_high_level_parameters_morphs_geometry() -> None:
    """The key coupling fix: set_high_level_parameters (the tool the agents
    already call) now deforms the wing, so no agent/prompt change is needed."""
    from tigl_mcp.tools.parameters import set_high_level_parameters_tool

    if not CPACS.exists():
        pytest.skip("D150 CPACS fixture not available")
    sm = SessionManager()
    session_id = open_cpacs_tool(sm).handler(
        {"source_type": "path", "source": str(CPACS)}
    )["session_id"]
    if sm.get(session_id)[1]._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")

    summary = get_wing_summary_tool(sm)
    before = summary.handler({"session_id": session_id, "wing_uid": WING})

    result = set_high_level_parameters_tool(sm).handler(
        {
            "session_id": session_id,
            "component_uid": WING,
            "updates": {"span": 45.0, "root_chord": 7.0, "tip_chord": 1.9, "sweep": 25.0},
        }
    )
    gm = result.get("geometry_morph")
    assert gm is not None and gm["rebuilt"] is True

    after = summary.handler({"session_id": session_id, "wing_uid": WING})
    # Geometry actually changed — the old behavior left this identical.
    assert abs(after["span"] - before["span"]) > 1.0


def test_morph_wing_requires_a_target() -> None:
    """No targets → a clear MorphError, not a silent no-op."""
    from tigl_mcp.errors import MCPError

    sm = SessionManager()
    session_id = open_cpacs_tool(sm).handler(
        {"source_type": "path", "source": str(CPACS)}
    )["session_id"]
    with pytest.raises(MCPError):
        morph_wing_tool(sm).handler({"session_id": session_id, "wing_uid": WING})
