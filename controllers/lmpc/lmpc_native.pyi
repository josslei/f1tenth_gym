from __future__ import annotations

from typing import Sequence

RACING_STATE_COLUMNS: tuple[str, str, str, str, str, str]
PAPER_STATE_COLUMNS: tuple[str, str, str, str, str, str]

class GymVehicleState:
    x: float
    y: float
    yaw: float
    v_x: float
    v_y: float
    omega: float
    def __init__(
        self, x: float, y: float, yaw: float, v_x: float, v_y: float, omega: float
    ) -> None: ...

class RacingLmpcState:
    s: float
    e_y: float
    e_psi: float
    v_x: float
    v_y: float
    omega: float
    def __init__(self) -> None: ...
    def to_array(self) -> list[float]: ...

class PaperLmpcState:
    v_x: float
    v_y: float
    omega: float
    e_psi: float
    s: float
    e_y: float
    def __init__(self) -> None: ...
    def to_array(self) -> list[float]: ...

class LmpcControlCommand:
    steering: float
    velocity: float
    def __init__(self) -> None: ...

class LmpcReference:
    curvature: float
    target_speed: float
    left_bound: float
    right_bound: float
    curvature_sequence: list[float]
    def __init__(self) -> None: ...

class LmpcConfig:
    horizon: int
    dt: float
    target_speed: float
    max_cpu_time: float
    max_iter: int
    tolerance: float
    track_half_width: float
    max_drive_force: float
    max_brake_force: float
    max_steer: float
    wheelbase: float
    track_length: float
    linearization_speed_floor: float
    max_lap_stored: int
    reg_dist_max: float
    reg_max_points: int
    reg_max_points_per_lap: int
    regression_horizon_stride: int
    lateral_weight: float
    heading_weight: float
    terminal_lateral_weight: float
    terminal_heading_weight: float
    input_weight_lon: float
    input_weight_steer: float
    control_rate_weight: float
    safe_set_cost_weight: float
    command_preview_steps: int
    def __init__(self) -> None: ...

class SparseErrorModel:
    A: list[list[float]]
    B: list[list[float]]
    C: list[float]

class FrenetProjection:
    s: float
    e_y: float
    heading: float
    segment_index: int

class CenterlineTrack:
    def __init__(
        self, x: Sequence[float], y: Sequence[float], closed: bool = True
    ) -> None: ...
    def project(self, x: float, y: float) -> FrenetProjection: ...
    def to_racing_state(self, state: GymVehicleState) -> RacingLmpcState: ...
    def to_paper_state(self, state: GymVehicleState) -> PaperLmpcState: ...
    def total_length(self) -> float: ...
    def s(self) -> list[float]: ...

def normalize_angle(angle: float) -> float: ...
def racing_to_paper(state: RacingLmpcState) -> PaperLmpcState: ...

class NativeLMPCController:
    def __init__(self, config: LmpcConfig | None = None) -> None: ...
    def reset(self) -> None: ...
    def update(self, state: RacingLmpcState) -> None: ...
    def set_reference(self, reference: LmpcReference) -> None: ...
    def add_initial_lap(
        self,
        x: Sequence[Sequence[float]],
        u: Sequence[Sequence[float]],
        k: Sequence[float],
        t: Sequence[float],
    ) -> None: ...
    def set_curvature_profile(
        self, s: Sequence[float], k: Sequence[float], total_length: float
    ) -> None: ...
    def control(self) -> LmpcControlCommand: ...
    def predicted_horizon(self) -> list[list[float]]: ...
    def error_model(self) -> SparseErrorModel: ...
    def sample_count(self) -> int: ...
    def completed_laps(self) -> int: ...
    def lap_sample_count(self) -> int: ...
    def last_safe_set_points(self) -> int: ...
    def solver_success_rate(self) -> float: ...
    def last_solver_status(self) -> str: ...
