"""Wing planform measurements from TiGL chord points: taper ratio and sweep angles.

TiGL has no sweep or taper query (``wingGetSweep`` does not exist, so ``sweep_deg`` was always
null), and a morph scales chords uniformly, so nothing reported either. Both follow directly from
the leading- and trailing-edge points TiGL does expose (``wingGetChordPoint`` at xsi 0 and 1).

Definitions (one half-wing, root section to tip section; x aft, spanwise = y for a wing or tailplane,
z for a vertical fin):

* ``root_chord_m`` / ``tip_chord_m``: |TE - LE| at the first section (eta 0 of segment 1) and the
  last section (eta 1 of the last segment).
* ``taper_ratio``: tip chord / root chord.
* ``sweep_le_deg``: leading-edge sweep of the straight line root LE -> tip LE.
* ``sweep_quarter_chord_deg``: sweep of the straight line root c/4 -> tip c/4. This is the single
  quarter-chord sweep of the equivalent straight-tapered wing, the quantity Aviary/FLOPS calls
  ``Aircraft.Wing.SWEEP``.
* ``panels``: the leading-edge and quarter-chord sweep of every segment (a kinked wing has several).

D150 Wing1 (measured 2026-10-03): root chord 6.076 m, tip chord 1.495 m, taper 0.246, LE sweep
24.9 deg, quarter-chord sweep 21.6 deg.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Point = Sequence[float]


def _chord(le: Point, te: Point) -> float:
    return math.dist(le, te)


def _quarter(le: Point, te: Point) -> tuple[float, float, float]:
    return tuple(a + 0.25 * (b - a) for a, b in zip(le, te))  # type: ignore[return-value]


def _sweep_deg(p0: Point, p1: Point, axis: int = 1) -> float | None:
    """Sweep of the line p0 -> p1: x is aft, ``axis`` is spanwise (1 = y for a wing, 2 = z for a fin)."""
    ds = abs(p1[axis] - p0[axis])
    if ds < 1e-9:
        return None
    return math.degrees(math.atan((p1[0] - p0[0]) / ds))


def _span_axis(root: Point, tip: Point) -> int:
    """y for a wing or tailplane, z for a vertical fin: whichever the wing extends along more."""
    return 1 if abs(tip[1] - root[1]) >= abs(tip[2] - root[2]) else 2


def planform_from_sections(sections: Sequence[tuple[Point, Point]]) -> dict[str, Any]:
    """Planform metrics from (leading edge, trailing edge) points ordered root -> tip.

    ``sections`` holds one (LE, TE) pair per section boundary: the root, every kink, the tip.
    Consecutive duplicates (a segment of zero span) are ignored for the panel sweeps.
    """
    if len(sections) < 2:
        raise ValueError("need at least the root and the tip section")
    root_le, root_te = sections[0]
    tip_le, tip_te = sections[-1]
    axis = _span_axis(root_le, tip_le)
    root_chord = _chord(root_le, root_te)
    tip_chord = _chord(tip_le, tip_te)
    panels = []
    for (le0, te0), (le1, te1) in zip(sections, sections[1:]):
        le_sweep = _sweep_deg(le0, le1, axis)
        if le_sweep is None:
            continue
        panels.append({
            "span_start_m": le0[axis], "span_end_m": le1[axis],
            "sweep_le_deg": le_sweep,
            "sweep_quarter_chord_deg": _sweep_deg(_quarter(le0, te0), _quarter(le1, te1), axis),
        })
    return {
        "root_chord_m": root_chord,
        "tip_chord_m": tip_chord,
        "taper_ratio": tip_chord / root_chord if root_chord > 0 else None,
        "sweep_le_deg": _sweep_deg(root_le, tip_le, axis),
        "sweep_quarter_chord_deg": _sweep_deg(_quarter(root_le, root_te), _quarter(tip_le, tip_te), axis),
        "span_axis": "y" if axis == 1 else "z",
        "panels": panels,
    }


def measure_planform(tigl: Any, wing_index: int) -> dict[str, Any] | None:
    """Read the section chord points of one wing from a TiGL handle and measure its planform.

    Returns None when the handle cannot answer (no segments, an exception), so callers keep their
    previous behaviour.
    """
    try:
        count = int(tigl.wingGetSegmentCount(wing_index))
        if count < 1:
            return None
        sections = [(tigl.wingGetChordPoint(wing_index, 1, 0.0, 0.0),
                     tigl.wingGetChordPoint(wing_index, 1, 0.0, 1.0))]
        for seg in range(1, count + 1):
            sections.append((tigl.wingGetChordPoint(wing_index, seg, 1.0, 0.0),
                             tigl.wingGetChordPoint(wing_index, seg, 1.0, 1.0)))
        return planform_from_sections(sections)
    except Exception:  # noqa: BLE001 - a measurement must never break the caller
        return None
