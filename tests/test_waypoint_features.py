import numpy as np

from utils.f110_env import F1TenthObservationConfig, build_observation, observation_dim
from utils.waypoint_utils import (
    closed_path_length,
    cumulative_arc_lengths,
    nearest_waypoint_index,
    project_to_closed_path,
    resample_path,
)


# ── resample_path ─────────────────────────────────────────────────────────────


class TestResamplePath:
    def test_straight_line(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        resampled = resample_path(path, spacing=0.5)
        assert resampled.shape[0] == 5  # 2.0 m / 0.5 m = 4 segments → 5 points
        assert np.allclose(resampled[:, 1], 0.0)
        assert np.isclose(resampled[-1, 0], 2.0)

    def test_l_shape(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        resampled = resample_path(path, spacing=0.5)
        # total length = 1.0 + 1.0 = 2.0 → 4 segments → 5 points
        assert resampled.shape[0] >= 4
        # first and last match input
        assert np.allclose(resampled[0], [0.0, 0.0])
        assert np.allclose(resampled[-1], [1.0, 1.0])

    def test_preserves_last_point(self):
        path = np.array([[0.0, 0.0], [0.3, 0.0], [0.7, 0.0]], dtype=np.float64)
        resampled = resample_path(path, spacing=0.5)
        assert np.allclose(resampled[-1], [0.7, 0.0])

    def test_single_segment_noop(self):
        path = np.array([[0.0, 0.0]], dtype=np.float64)
        resampled = resample_path(path, spacing=0.5)
        assert resampled.shape == (1, 2)
        assert np.allclose(resampled[0], [0.0, 0.0])


class TestNearestWaypointIndex:
    def test_default_start_scans_full_track(self):
        waypoints = np.stack(
            [np.arange(300, dtype=np.float64), np.zeros(300, dtype=np.float64)],
            axis=1,
        )

        assert nearest_waypoint_index(waypoints, np.array([125.2, 0.0])) == 125


class TestClosedPathGeometry:
    def test_length_includes_closing_segment(self):
        path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]])

        assert np.isclose(closed_path_length(path), 3.0 + np.sqrt(5.0))

    def test_projection_uses_closing_segment(self):
        path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        path_s = cumulative_arc_lengths(path)
        midpoint = 0.5 * (path[-1] + path[0])

        s, ey, heading = project_to_closed_path(path, path_s, midpoint, 0)

        assert np.isclose(s, 3.0 + 0.5 * np.sqrt(5.0))
        assert np.isclose(ey, 0.0)
        assert np.isclose(heading, np.arctan2(-1.0, -2.0))


# ── observation_dim with waypoints ────────────────────────────────────────────


class TestObservationDimWithWaypoints:
    def test_without_waypoints_unchanged(self):
        config = F1TenthObservationConfig(
            scan_size=4,
            include_ego_state=True,
            include_waypoints=False,
        )
        assert observation_dim(config) == 4 + 7 + 2  # 7 ego-state + 2 prev_action

    def test_with_waypoints_adds_2_per_lookahead(self):
        config = F1TenthObservationConfig(
            scan_size=4,
            include_waypoints=True,
            lookahead_distances=(0.5, 1.0, 2.0),
        )
        # 4 scan + 7 ego + 3*2 waypoints + 2 prev_action = 19
        assert observation_dim(config) == 4 + 7 + 6 + 2

    def test_backward_compat(self):
        default = F1TenthObservationConfig()
        assert default.include_waypoints is False
        assert observation_dim(default) == 108 + 7 + 2  # no extra waypoint dim


# ── build_observation with waypoints ──────────────────────────────────────────


def _make_reset_obs(num_agents: int = 1) -> dict:
    return {
        "ego_idx": 0,
        "scans": np.zeros((num_agents, 1080), dtype=np.float64),
        "poses_x": np.zeros(num_agents, dtype=np.float64),
        "poses_y": np.zeros(num_agents, dtype=np.float64),
        "poses_theta": np.zeros(num_agents, dtype=np.float64),
        "linear_vels_x": np.zeros(num_agents, dtype=np.float64),
        "linear_vels_y": np.zeros(num_agents, dtype=np.float64),
        "ang_vels_z": np.zeros(num_agents, dtype=np.float64),
        "steer_angle": np.zeros(num_agents, dtype=np.float64),
        "collisions": np.zeros(num_agents, dtype=np.float64),
    }


class TestBuildObservationWithWaypoints:
    def test_without_waypoints_returns_correct_shape(self):
        config = F1TenthObservationConfig(
            scan_size=4,
            include_waypoints=False,
            lookahead_distances=(),
        )
        obs = _make_reset_obs()
        result = build_observation(obs, config)
        assert result.shape == (4 + 7 + 2,)

    def test_with_waypoints_appended(self):
        wp = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        )
        config = F1TenthObservationConfig(
            scan_size=4,
            include_waypoints=True,
            lookahead_distances=(0.5, 1.0),
            waypoint_resample_spacing=0.5,
            waypoint_scale=30.0,
            _waypoints=wp,
        )
        obs = _make_reset_obs()
        result = build_observation(obs, config)
        # 4 scan + 7 ego + 2*2 waypoints + 2 prev_action = 17
        assert result.shape == (4 + 7 + 4 + 2,)
        # Car at (0,0,0), nearest is (0,0), offset 1 → (0.5,0), offset 2 → (1.0,0)
        # x_rel/30.0: 0.5/30 ≈ 0.0167, then 1.0/30 ≈ 0.0333.  Prev_action zeros at end.
        assert np.isclose(result[-6], 0.5 / 30.0)
        assert np.isclose(result[-5], 0.0)
        assert np.isclose(result[-4], 1.0 / 30.0)
        assert np.isclose(result[-3], 0.0)
        assert np.isclose(result[-2], 0.0)
        assert np.isclose(result[-1], 0.0)

    def test_without_waypoints_when_internal_none(self):
        """include_waypoints=True but no _waypoints set → no crash, just no wp features."""
        config = F1TenthObservationConfig(
            scan_size=4,
            include_waypoints=True,
            lookahead_distances=(0.5,),
            _waypoints=None,
        )
        obs = _make_reset_obs()
        result = build_observation(obs, config)
        assert result.shape == (4 + 7 + 2 + 2,)
        assert np.allclose(result[-2:], [0.0, 0.0])
