from typing import Final

import numpy as np
from numpy.typing import NDArray

STATE_COLUMNS: Final[tuple[str, ...]]
DEFAULT_PARAMS: Final[F110Params]

class Integrator:
    Euler: Integrator
    RK4: Integrator

class F110Params:
    mu: float
    C_Sf: float
    C_Sr: float
    lf: float
    lr: float
    h: float
    m: float
    inertia: float
    s_min: float
    s_max: float
    sv_min: float
    sv_max: float
    v_switch: float
    a_max: float
    v_min: float
    v_max: float
    timestep: float
    def __init__(self) -> None: ...

class F110State:
    x: float
    y: float
    steer_angle: float
    velocity: float
    yaw_angle: float
    yaw_rate: float
    slip_angle: float
    steer_buffer_0: float
    steer_buffer_1: float
    steer_buffer_len: int
    in_collision: bool
    def __init__(self) -> None: ...
    def as_array(self) -> NDArray[np.float64]: ...

class F110Action:
    steer: float
    velocity: float
    def __init__(self, steer: float = 0.0, velocity: float = 0.0) -> None: ...

class F110StepResult:
    state: F110State
    reward: float
    discount: float
    terminal: bool

class TrackMap:
    height: int
    width: int
    resolution: float
    orig_x: float
    orig_y: float
    orig_c: float
    orig_s: float
    dt: list[float]
    theta_dis: int
    num_beams: int
    fov: float
    max_range: float
    eps: float
    ttc_thresh: float
    sines: list[float]
    cosines: list[float]
    side_distances: list[float]
    def __init__(self) -> None: ...
    def compute_scan_tables(self) -> None: ...
    def distance_at(self, x: float, y: float) -> float: ...

class F110ProgressReward:
    def __init__(
        self,
        track_map: TrackMap = ...,
        waypoints_x: list[float] | NDArray[np.float64] = ...,
        waypoints_y: list[float] | NDArray[np.float64] = ...,
        q_progress: float = 1.0,
        q_alpha: float = 1.0,
        q_smooth: float = 0.0,
        terminal_penalty: float = 1000.0,
        alpha_th: float = 0.0,
        slip_terminal_penalty: float = 0.0,
        q_offtrack_grad: float = 0.0,
    ) -> None: ...
    def set_waypoints(
        self,
        waypoints_x: list[float] | NDArray[np.float64],
        waypoints_y: list[float] | NDArray[np.float64],
    ) -> None: ...
    def reset(self) -> None: ...
    def __call__(
        self,
        px: float,
        py: float,
        theta: float,
        vx: float,
        vy: float,
        action_0: float,
        action_1: float,
        collision: bool,
        terminated: bool,
    ) -> float: ...

def step(
    state: F110State,
    action: F110Action,
    params: F110Params | None = None,
    integrator: Integrator = Integrator.RK4,
    direct_accel_control: bool = False,
) -> F110StepResult: ...
def step_batch(
    states: NDArray[np.float64],
    actions: NDArray[np.float64],
    params: F110Params | None = None,
    integrator: Integrator = Integrator.RK4,
) -> NDArray[np.float64]: ...
