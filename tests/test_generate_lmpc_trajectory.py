from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_lmpc_trajectory import (
    TRAJECTORY_COLUMNS,
    build_lmpc_trajectory_table,
)


def _params() -> dict:
    return {
        "stepsize_opts": {"stepsize_interp_after_opt": 0.5},
        "veh_params": {"v_max": 4.0, "g": 10.0},
        "optim_opts_mintime": {
            "mue": 0.5,
            "ax_pos_safe": None,
            "ax_neg_safe": None,
            "ay_safe": None,
        },
    }


def _square_track() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_lmpc_trajectory_table_has_upstream_column_layout() -> None:
    table = build_lmpc_trajectory_table(
        _square_track(), _params(), spacing=0.5, speed_mode="constant"
    )

    assert TRAJECTORY_COLUMNS == (
        "PX",
        "PY",
        "PZ",
        "YAW",
        "SPEED",
        "CURVATURE",
        "DIST_TO_SF_BWD",
        "DIST_TO_SF_FWD",
        "REGION",
        "LEFT_BOUND_X",
        "LEFT_BOUND_Y",
        "RIGHT_BOUND_X",
        "RIGHT_BOUND_Y",
        "BANK",
        "LON_ACC",
        "LAT_ACC",
        "TIME",
    )
    assert table.shape == (8, 17)
    assert table[:, 2] == pytest.approx(0.0)
    assert table[:, 4] == pytest.approx(4.0)
    assert table[:, 6] == pytest.approx(np.arange(0.0, 4.0, 0.5))
    assert table[:, 7] == pytest.approx(4.0 - table[:, 6])
    assert table[:, 8] == pytest.approx(0.0)
    assert table[:, 13] == pytest.approx(0.0)
    assert np.all(np.diff(table[:, 16]) > 0.0)


def test_curvature_speed_profile_uses_raceline_config_limits() -> None:
    table = build_lmpc_trajectory_table(
        _square_track(), _params(), spacing=0.5, speed_mode="curvature"
    )

    ay_limit = _params()["optim_opts_mintime"]["mue"] * _params()["veh_params"]["g"]
    pointwise_limit = np.minimum(
        _params()["veh_params"]["v_max"],
        np.sqrt(ay_limit / (np.abs(table[:, 5]) + 1.0e-6)),
    )
    assert np.all(table[:, 4] <= pointwise_limit + 1.0e-12)


def test_curvature_speed_profile_respects_acceleration_limits() -> None:
    params = _params()
    params["optim_opts_mintime"]["ax_pos_safe"] = 0.5
    params["optim_opts_mintime"]["ax_neg_safe"] = 0.5
    table = build_lmpc_trajectory_table(
        _square_track(), params, spacing=0.5, speed_mode="curvature"
    )

    speeds = table[:, 4]
    segment_lengths = np.r_[np.diff(table[:, 6]), table[0, 7] - table[-1, 6]]
    next_speeds = np.roll(speeds, -1)
    accel = (next_speeds * next_speeds - speeds * speeds) / (2.0 * segment_lengths)
    assert np.all(accel <= 0.5 + 1.0e-12)
    assert np.all(accel >= -0.5 - 1.0e-12)


def test_ay_safe_overrides_friction_based_lateral_limit() -> None:
    params = _params()
    params["optim_opts_mintime"]["ay_safe"] = 1.0

    table = build_lmpc_trajectory_table(
        _square_track(), params, spacing=0.5, speed_mode="curvature"
    )

    pointwise_limit = np.minimum(4.0, np.sqrt(1.0 / (np.abs(table[:, 5]) + 1.0e-6)))
    assert np.all(table[:, 4] <= pointwise_limit + 1.0e-12)
