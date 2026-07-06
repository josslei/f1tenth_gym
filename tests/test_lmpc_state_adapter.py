from __future__ import annotations

import math

import numpy as np
import pytest

from controllers.controller_base import Controller
from controllers.controller_base import VehicleState
from controllers.lmpc import LMPCController
import controllers.lmpc.controller as controller_module

lmpc = pytest.importorskip("controllers.lmpc.binding")
adapter = pytest.importorskip("controllers.lmpc.adapter")


def test_state_column_orders_are_explicit() -> None:
    assert lmpc.RACING_STATE_COLUMNS == ("s", "e_y", "e_psi", "v_x", "v_y", "omega")
    assert lmpc.PAPER_STATE_COLUMNS == ("v_x", "v_y", "omega", "e_psi", "s", "e_y")


def test_lmpc_controller_uses_controller_contract() -> None:
    assert issubclass(LMPCController, Controller)


def test_lmpc_controller_projects_centerline_before_native_update(monkeypatch) -> None:
    class NativeCommand:
        steering = 0.1
        velocity = 2.0

    class NativeController:
        def __init__(self, config) -> None:
            self.config = config
            self.state = None
            self.reset_called = False

        def reset(self) -> None:
            self.reset_called = True

        def update(self, state) -> None:
            self.state = state

        def set_reference(self, reference) -> None:
            self.reference = reference

        def control(self) -> NativeCommand:
            return NativeCommand()

        def sample_count(self) -> int:
            return 12

        def completed_laps(self) -> int:
            return 1

        def lap_sample_count(self) -> int:
            return 5

        def last_safe_set_points(self) -> int:
            return 7

    monkeypatch.setattr(controller_module, "NativeLMPCController", NativeController)
    controller = LMPCController(
        np.array([0.0, 10.0]),
        np.array([0.0, 0.0]),
        False,
        regression_horizon_stride=4,
    )
    native = controller.native_controller
    controller.update(
        VehicleState(3.0, 2.0, 0.1, 4.0), lateral_velocity=0.5, yaw_rate=0.2
    )
    command = controller.control()

    assert native.state is not None
    assert native.config.regression_horizon_stride == 8
    assert native.config.track_length == pytest.approx(10.0)
    assert native.state.to_array() == pytest.approx([3.0, 2.0, 0.1, 4.0, 0.5, 0.2])
    assert command.steering == pytest.approx(0.1)
    assert command.velocity == pytest.approx(2.0)
    assert controller.sample_count() == 12
    assert controller.completed_laps() == 1
    assert controller.lap_sample_count() == 5
    assert controller.last_safe_set_points() == 7


def test_lmpc_controller_loads_upstream_trajectory_table(monkeypatch, tmp_path) -> None:
    class NativeCommand:
        steering = 0.1
        velocity = 2.0

    class NativeController:
        def __init__(self, config) -> None:
            self.config = config
            self.state = None

        def reset(self) -> None: ...

        def update(self, state) -> None:
            self.state = state

        def set_reference(self, reference) -> None:
            self.reference = reference

        def control(self) -> NativeCommand:
            return NativeCommand()

    table = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                10.0,
                0.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                10.0,
                0.0,
                0.0,
                0.0,
                5.0,
                0.2,
                10.0,
                0.0,
                0.0,
                10.0,
                1.0,
                10.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                2.0,
            ],
        ],
        dtype=np.float64,
    )
    table_path = tmp_path / "trajectory.txt"
    np.savetxt(table_path, table)

    monkeypatch.setattr(controller_module, "NativeLMPCController", NativeController)
    controller = LMPCController.from_trajectory_table(table_path)
    native = controller.native_controller
    controller.update(VehicleState(5.0, 0.0, 0.0, 4.0))
    command = controller.control()

    assert native.config.target_speed == pytest.approx(4.0)
    assert native.config.regression_horizon_stride == 8
    assert native.config.track_length == pytest.approx(10.0)
    assert native.state.to_array() == pytest.approx([5.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    assert native.reference.target_speed == pytest.approx(4.0)
    assert native.reference.curvature == pytest.approx(0.1)
    assert len(native.reference.curvature_sequence) == native.config.horizon - 1
    assert native.reference.curvature_sequence[0] == pytest.approx(0.1)
    assert native.reference.left_bound == pytest.approx(1.0)
    assert native.reference.right_bound == pytest.approx(1.0)
    assert command.steering == pytest.approx(0.1)
    assert command.velocity == pytest.approx(2.0)


def test_lmpc_controller_updates_from_full_gym_observation(monkeypatch) -> None:
    class NativeController:
        def __init__(self, config) -> None:
            self.state = None

        def reset(self) -> None: ...

        def update(self, state) -> None:
            self.state = state

        def set_reference(self, reference) -> None: ...

        def control(self): ...

    monkeypatch.setattr(controller_module, "NativeLMPCController", NativeController)
    controller = LMPCController([0.0, 10.0], [0.0, 0.0], False)
    native = controller.native_controller
    controller.update_from_observation(
        {
            "ego_idx": 0,
            "poses_x": [3.0],
            "poses_y": [2.0],
            "poses_theta": [0.1],
            "linear_vels_x": [4.0],
            "linear_vels_y": [0.5],
            "ang_vels_z": [0.2],
        }
    )

    assert native.state is not None
    assert native.state.to_array() == pytest.approx([3.0, 2.0, 0.1, 4.0, 0.5, 0.2])


def test_centerline_projection_and_state_conversion() -> None:
    track = lmpc.CenterlineTrack([0.0, 10.0], [0.0, 0.0], False)
    state = lmpc.GymVehicleState(3.0, 2.0, 0.1, 4.0, 0.5, 0.2)

    projection = track.project(state.x, state.y)
    racing_state = track.to_racing_state(state)
    paper_state = track.to_paper_state(state)

    assert projection.segment_index == 0
    assert projection.s == pytest.approx(3.0)
    assert projection.e_y == pytest.approx(2.0)
    assert projection.heading == pytest.approx(0.0)

    assert racing_state.to_array() == pytest.approx([3.0, 2.0, 0.1, 4.0, 0.5, 0.2])
    assert paper_state.to_array() == pytest.approx([4.0, 0.5, 0.2, 0.1, 3.0, 2.0])


def test_angle_normalization() -> None:
    assert lmpc.normalize_angle(3.5 * math.pi) == pytest.approx(-0.5 * math.pi)


def test_gym_observation_adapter() -> None:
    state = adapter.obs_to_gym_vehicle_state(
        {
            "ego_idx": 1,
            "poses_x": [0.0, 1.0],
            "poses_y": [0.0, 2.0],
            "poses_theta": [0.0, 0.3],
            "linear_vels_x": [0.0, 4.0],
            "linear_vels_y": [0.0, 0.5],
            "ang_vels_z": [0.0, 0.2],
        }
    )

    assert state.x == pytest.approx(1.0)
    assert state.y == pytest.approx(2.0)
    assert state.yaw == pytest.approx(0.3)
    assert state.v_x == pytest.approx(4.0)
    assert state.v_y == pytest.approx(0.5)
    assert state.omega == pytest.approx(0.2)
