"""B115: the mesh must be the WHOLE aircraft, not the half CPACS literally defines.

CPACS models a symmetric aircraft as one half plus ``symmetry="x-z-plane"``. The legacy
``exportFusedBREP`` writes exactly that half, so every CFD solve run on the 7960 saw a full
fuselage, a starboard wing and a starboard tailplane, with nothing on the port side and no
symmetry-plane boundary condition. Half the lift at the full wing's reference area, an asymmetric
body and a sealed wing root: CL 0.05-0.22 where cruise needs ~0.57, and a converged, stable,
NEGATIVE CD in 5 of 16 links of geoauth_all8_7960.
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

from tigl_mcp.tools import volume_mesh

CPACS = Path(__file__).parent / "fixtures" / "D150_simple.xml"


# ---- the option is asked for, and a failure falls back rather than breaking the run -------------

class _Options:
    def __init__(self):
        self.symmetries = None
        self.farfield = None

    def set_apply_symmetries(self, value):
        self.symmetries = value

    def set_include_farfield(self, value):
        self.farfield = value


class _Exporter:
    def __init__(self, options):
        self.options = options
        self.added = []

    def add_fused_configuration(self, aircraft):
        self.added.append(aircraft)

    def write(self, path):
        Path(path).write_bytes(b"BREP-CONTENT")


def _fake_exports(monkeypatch, exporter_cls=_Exporter):
    made = {}

    def _factory(options):
        made["exporter"] = exporter_cls(options)
        return made["exporter"]

    module = types.SimpleNamespace(BRepOptions=_Options, CTiglExportBrep=_factory)
    monkeypatch.setitem(sys.modules, "tigl3", types.SimpleNamespace(exports=module))
    monkeypatch.setitem(sys.modules, "tigl3.exports", module)
    monkeypatch.setattr(volume_mesh, "_tigl3_aircraft", lambda handle: "aircraft")
    return made


def test_the_export_asks_for_symmetries(monkeypatch):
    made = _fake_exports(monkeypatch)
    out = volume_mesh._export_fused_brep_with_symmetries(object())
    assert out == b"BREP-CONTENT"
    assert made["exporter"].options.symmetries is True
    assert made["exporter"].options.farfield is False
    assert made["exporter"].added == ["aircraft"]


def test_no_tigl3_configuration_falls_back(monkeypatch):
    monkeypatch.setattr(volume_mesh, "_tigl3_aircraft", lambda handle: None)
    assert volume_mesh._export_fused_brep_with_symmetries(object()) is None


def test_a_failing_writer_falls_back_instead_of_raising(monkeypatch):
    class _Boom(_Exporter):
        def write(self, path):
            raise RuntimeError("exporter exploded")

    _fake_exports(monkeypatch, _Boom)
    assert volume_mesh._export_fused_brep_with_symmetries(object()) is None


def test_the_configuration_export_prefers_the_symmetric_one(monkeypatch):
    monkeypatch.setattr(volume_mesh, "_export_fused_brep_with_symmetries", lambda h: b"WHOLE")

    class _Handle:
        _tigl_handle = types.SimpleNamespace(exportFusedBREP=lambda p: None)

    assert volume_mesh._export_configuration_brep(_Handle()) == b"WHOLE"


# ---- and the geometry really does come back whole ------------------------------------------------

pytest.importorskip("tigl3")


@pytest.mark.integration_real
def test_the_exported_aircraft_spans_both_sides():
    """D150 baseline: -1.98..16.96 in y without symmetries, -16.96..16.96 with them."""
    if not CPACS.exists():
        pytest.skip("D150 CPACS fixture not available")
    import tigl3.tigl3wrapper as t3
    import tixi3.tixi3wrapper as tx3
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Shape

    from tigl3 import configuration, exports

    tixi = tx3.Tixi3()
    tixi.open(str(CPACS))
    tigl = t3.Tigl3()
    tigl.open(tixi, "")
    aircraft = configuration.CCPACSConfigurationManager_get_instance().get_configuration(
        tigl._handle.value)

    spans = {}
    for apply_symmetries in (False, True):
        options = exports.BRepOptions()
        options.set_apply_symmetries(apply_symmetries)
        options.set_include_farfield(False)
        exporter = exports.CTiglExportBrep(options)
        exporter.add_fused_configuration(aircraft)
        path = tempfile.mktemp(suffix=".brep")
        exporter.write(path)
        shape, builder, box = TopoDS_Shape(), BRep_Builder(), Bnd_Box()
        breptools.Read(shape, path, builder)
        brepbndlib.Add(shape, box)
        _x0, y0, _z0, _x1, y1, _z1 = box.Get()
        spans[apply_symmetries] = (y0, y1)

    half_y0, half_y1 = spans[False]
    full_y0, full_y1 = spans[True]
    assert half_y0 > -5.0, "without symmetries CPACS defines only the starboard half"
    assert full_y0 == pytest.approx(-full_y1, rel=1e-3), "the aircraft must be symmetric in y"
    assert full_y1 == pytest.approx(half_y1, rel=1e-3), "the starboard side must be unchanged"
