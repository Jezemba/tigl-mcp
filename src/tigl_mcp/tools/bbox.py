"""Tight axis-aligned box of an OCC shape (Avion J2, 2026-11-18).

``brepbndlib.Add`` (what tigl-mcp used) returns a box enlarged by the shape's tolerance and by the B-spline control
polygons: for the D150 fuselage 37.6843 x 3.9597 x 4.1490 m, while the surface spans 37.6446 x 3.9518 x 4.1437 m
(dense TiGL surface points 37.6445 x 3.9518 x 4.1431; ws8 /data/Jessica/final_fix/j1j2/out/fus_probe_tigl-env.txt).
Aviary flew that fuselage 4 cm longer than mass-mcp sized it (-63 kg GTOW of the J2 gap). ``AddOptimal`` without
triangulation and without shape tolerance bounds the exact surfaces.
"""

from __future__ import annotations

from typing import Any


def tight_box(shape: Any) -> tuple[float, float, float, float, float, float]:
    """(xmin, ymin, zmin, xmax, ymax, zmax) of the exact surfaces of ``shape``."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, False, False)
    return box.Get()
