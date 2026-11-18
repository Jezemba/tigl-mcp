"""Avion J2 (2026-11-18): measured boxes are the exact surfaces', not brepbndlib.Add's enlarged ones.

D150 (ws8 /data/Jessica/final_fix/j1j2/out/fus_probe_tigl-env.txt, bbox_probe_tigl-env.json): the fuselage surface
spans 37.6445 x 3.9518 x 4.1431 m (dense TiGL points); brepbndlib.Add gave 37.6843 x 3.9597 x 4.1490 m, so aviary
flew a fuselage 4 cm longer than mass-mcp sized it. Wing boxes were enlarged by up to 3.5 mm (fin thickness).
Needs the native TiGL runtime (skipped without it). No network, no live server.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tigl_mcp.session_manager import SessionManager
from tigl_mcp.tools import build_tools

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"
FUSELAGE_SURFACE_M = (37.6445, 3.9518, 4.1431)   # dense TiGL surface points (41 x 721 per segment)
LOOSE_FUSELAGE_BOX_M = (37.6843, 3.9597, 4.1490)  # what brepbndlib.Add reported


def _d150():
    pytest.importorskip("tigl3")
    pytest.importorskip("OCC.Core.BRepBndLib")
    if not CPACS.exists():
        pytest.skip("D150 fixture not available")
    sm = SessionManager()
    tools = build_tools(sm)
    open_cpacs = next(t for t in tools if t.name == "open_cpacs")
    sid = open_cpacs.handler({"source_type": "path", "source": str(CPACS)})["session_id"]
    _, handle, config = sm.get(sid)
    if handle._tigl_handle is None:
        pytest.skip("native TiGL runtime not available")
    return handle._tigl_handle, config


def _wing_points_extent(native, index):
    pts = []
    for seg in range(1, native.wingGetSegmentCount(index) + 1):
        for eta in np.linspace(0.0, 1.0, 11):
            for xsi in np.linspace(0.0, 1.0, 41):
                pts.append(native.wingGetUpperPoint(index, seg, eta, xsi))
                pts.append(native.wingGetLowerPoint(index, seg, eta, xsi))
    pts = np.array(pts)
    return pts.max(0) - pts.min(0)


@pytest.mark.integration_real
def test_fuselage_dims_are_the_surface_to_a_millimetre():
    from tigl_mcp.tools.morph import _fuselage_dims

    native, _ = _d150()
    d = _fuselage_dims(native, "Fuselage1")
    for got, want, loose in zip((d["length"], d["width"], d["height"]), FUSELAGE_SURFACE_M, LOOSE_FUSELAGE_BOX_M):
        assert got == pytest.approx(want, abs=1e-3)
        assert got < loose - 4e-3          # the enlarged box is gone


@pytest.mark.integration_real
def test_component_boxes_are_the_surfaces_to_a_millimetre():
    from tigl_mcp.tools.configuration import _measured_bbox

    native, config = _d150()
    for comp in config.all_components():
        box = _measured_bbox(native, comp)
        assert box is not None, comp.uid
        ext = np.array([box["xmax"] - box["xmin"], box["ymax"] - box["ymin"], box["zmax"] - box["zmin"]])
        if (comp.type_name or "").lower() == "fuselage":
            assert ext == pytest.approx(np.array(FUSELAGE_SURFACE_M), abs=1e-3)
            continue
        pts = _wing_points_extent(native, comp.index)
        if str(comp.symmetry or "").lower() not in ("", "0", "none"):   # mirrored about x-z: tip to tip
            assert box["ymax"] == pytest.approx(-box["ymin"], abs=1e-9)        # mirrored box
            assert box["ymax"] == pytest.approx(native.wingGetSpan(comp.uid) / 2.0, abs=1e-3)
            ext, pts = ext[[0, 2]], pts[[0, 2]]
        assert ext == pytest.approx(pts, abs=1e-3), comp.uid
