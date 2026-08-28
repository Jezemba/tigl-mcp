"""An agent-supplied mesh_size_min must not be able to take the server down.

Measured 2026-08-20 on the D150 fixture (characteristic length ~17 m, calibrated
default mesh_size_min 0.30):

    mesh_size_min=0.5  ->  55 s, 114 MB mesh
    mesh_size_min=0.1  ->  still running at 21 MINUTES, RSS climbing linearly
                           2.2 -> 14.7 GB, killed manually

Cell count scales roughly with (1/h)^3, so a threefold-finer request is ~27x the
mesh. That mattered far beyond one slow call: the handler is SYNCHRONOUS, so while
it runs tigl-mcp answers nothing -- it is indistinguishable from a dead server.
The outage outlived the run and silently invalidated 6 of 16 runs in the
2026-08-18 sweep, several still recorded as `success` (B69).
"""

import pytest

from tigl_mcp.tools.volume_mesh import MESH_SIZE_MIN_FLOOR_FRACTION

CHAR_LENGTH = 16.956361853508927   # D150 fixture, from a real run's stats
DEFAULT = round(CHAR_LENGTH * 0.0176, 4)          # 0.2984 -- the calibrated value
FLOOR = round(CHAR_LENGTH * MESH_SIZE_MIN_FLOOR_FRACTION, 4)


def clamp(requested):
    """Mirror of the clamp in _generate_volume_mesh_gmsh."""
    return max(requested, FLOOR) if requested is not None else DEFAULT


class TestTheFloorIsWhereItShouldBe:
    def test_floor_is_half_the_calibrated_default(self):
        assert FLOOR == pytest.approx(DEFAULT / 2, rel=0.01)

    def test_floor_is_finer_than_the_default_not_coarser(self):
        """The clamp must not degrade the recommended mesh."""
        assert FLOOR < DEFAULT


class TestTheRunawayValueIsBounded:
    def test_the_value_that_took_the_server_down_is_clamped(self):
        """0.1 is what the agent actually asked for."""
        assert clamp(0.1) == FLOOR
        assert clamp(0.1) > 0.1

    @pytest.mark.parametrize("runaway", [0.1, 0.05, 0.01, 0.001])
    def test_ever_finer_requests_all_land_on_the_floor(self, runaway):
        assert clamp(runaway) == FLOOR


class TestLegitimateRequestsArePreserved:
    def test_a_coarser_request_is_untouched(self):
        """0.5 completed in 55 s -- there is no reason to alter it."""
        assert clamp(0.5) == 0.5

    def test_the_calibrated_default_is_untouched(self):
        assert clamp(DEFAULT) == DEFAULT

    def test_omitting_it_still_gives_the_calibrated_default(self):
        assert clamp(None) == DEFAULT

    def test_a_value_just_above_the_floor_is_untouched(self):
        assert clamp(FLOOR + 0.01) == FLOOR + 0.01


class TestTheFloorScalesWithGeometry:
    """A morphed wing changes characteristic length; the floor must follow it,
    or the clamp would be wrong for any geometry but the D150."""

    @pytest.mark.parametrize("char_len", [5.0, 17.0, 50.0])
    def test_floor_tracks_characteristic_length(self, char_len):
        floor = char_len * MESH_SIZE_MIN_FLOOR_FRACTION
        default = char_len * 0.0176
        assert floor < default
        assert floor == pytest.approx(default / 2, rel=0.01)


class TestTheCostCapCatchesWhatTheClampMissed:
    """Clamping mesh_size_min alone was NOT enough.

    Measured 2026-08-27, AFTER the clamp shipped: an agent reached the same hang
    through the other two knobs --

        mesh_size_min=0.1 (clamped to 0.1492), mesh_size_max=2.0, far_field_distance=15.0

    A 3x larger domain meshed ~4x more finely at the boundary. tigl stopped
    answering for 15+ minutes and the sweep was lost. Cost depends on the DOMAIN
    and BOTH size bounds, so a single-parameter clamp can always be routed around.
    """

    from tigl_mcp.tools.volume_mesh import MAX_ESTIMATED_CELLS as CAP

    @staticmethod
    def estimate(char_length, far_field_distance, size_min, size_max):
        """Mirror of the pre-flight estimate in _generate_volume_mesh_gmsh.

        Two terms because gmsh meshes fine near the body and coarse in the far
        field, and either can dominate. A single-term version was degenerate --
        it returned 1,000,000 for the calibrated default AND for the 21-minute
        runaway, so it caught nothing.
        """
        edge = 2 * far_field_distance * char_length
        return (edge / size_max) ** 3 + (char_length / size_min) ** 3

    def test_the_calibrated_default_passes_comfortably(self):
        est = self.estimate(CHAR_LENGTH, 10.0, DEFAULT, round(CHAR_LENGTH * 0.47, 3))
        assert est < self.CAP

    def test_the_request_that_hung_the_server_is_refused(self):
        """The exact arguments from the lost run."""
        est = self.estimate(CHAR_LENGTH, 15.0, FLOOR, 2.0)
        assert est > self.CAP

    def test_the_earlier_runaway_shape_is_also_refused(self):
        """far_field 5.0 with an unclamped 0.1 -- the original 21-minute call."""
        est = self.estimate(CHAR_LENGTH, 5.0, 0.1, 2.0)
        assert est > self.CAP

    def test_a_moderate_finer_request_is_still_allowed(self):
        """The cap must not make legitimate refinement impossible."""
        est = self.estimate(CHAR_LENGTH, 10.0, FLOOR, round(CHAR_LENGTH * 0.47, 3))
        assert est < self.CAP

    def test_the_estimate_tracks_the_observed_default(self):
        """The calibrated default really does produce ~112k-350k cells."""
        est = self.estimate(CHAR_LENGTH, 10.0, DEFAULT, round(CHAR_LENGTH * 0.47, 3))
        assert 100_000 < est < 500_000

    def test_far_field_dominates_the_bulk_term_cubically(self):
        """Why the error tells the agent to reduce far_field_distance FIRST."""
        a = (2 * 5.0 * CHAR_LENGTH / 8.0) ** 3
        b = (2 * 10.0 * CHAR_LENGTH / 8.0) ** 3
        assert b / a == pytest.approx(8.0, rel=0.01)
