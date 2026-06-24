from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from typing import overload

from numpy.typing import NDArray

class F110Params:
    mu: float
    c_sf: float
    c_sr: float
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

class F110State:
    x: float
    y: float
    steer_angle: float
    velocity: float
    yaw_angle: float
    yaw_rate: float
    slip_angle: float

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
    def compute_scan_tables(self) -> None: ...

class ObservationConfig:
    scan_size: int
    scan_max_m: float
    include_ego_state: bool
    speed_scale: float
    yaw_rate_scale: float
    steer_scale: float
    include_waypoints: bool
    lookahead_distances: list[float]
    waypoint_scale: float
    waypoint_resample_spacing: float

class ActionLattice:
    @overload
    def __init__(
        self,
        steering_bins: int,
        velocity_bins: int,
        velocity_min: float = 1.0,
        velocity_max: float = 8.0,
    ) -> None: ...
    @overload
    def __init__(
        self,
        steering_bins: np.ndarray | list[float],
        velocity_bins: np.ndarray | list[float],
    ) -> None: ...
    @property
    def action_count(self) -> int: ...
    def normalized_action(self, action_index: int) -> NDArray[np.float32]: ...
    def normalized_batch(
        self, action_indices: NDArray[np.int64]
    ) -> NDArray[np.float32]: ...

class MuZeroSearchAdapter:
    def __init__(
        self,
        model_path: str,
        num_iters: int,
        temperature: float,
        c_puct: float,
        batch_size: int,
        action_count: int,
        hidden_size: int,
        max_nodes: int = 0,
        device: str = "",
        print_metrics: bool = False,
        *,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None: ...
    def search_batch(self, obs_batch: np.ndarray) -> SimpleNamespace: ...

class SelfPlayEngine:
    def __init__(
        self,
        search: MuZeroSearchAdapter,
        track_map: TrackMap,
        obs_config: ObservationConfig,
        action_lattice: ActionLattice,
        discount: float,
        sample_actions: bool,
        print_metrics: bool,
        waypoints_x: np.ndarray | list[float],
        waypoints_y: np.ndarray | list[float],
        cum_arc_lengths: np.ndarray | list[float],
        dynamics_params: F110Params,
        car_length: float,
        car_width: float,
        q_progress: float,
        q_alpha: float,
        q_smooth: float,
        terminal_penalty: float,
        alpha_th: float,
        slip_terminal_penalty: float,
        q_offtrack_grad: float,
    ) -> None: ...
    def generate(
        self,
        rollout_steps: int,
        batch_size: int,
        initial_states: np.ndarray,
    ) -> SimpleNamespace: ...

class SelfPlayResult:
    trajectories: list[SimpleNamespace]
    metrics: dict[str, float]
