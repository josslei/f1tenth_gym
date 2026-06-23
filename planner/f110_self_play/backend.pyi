from __future__ import annotations

import numpy as np
import torch

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

class ActionLattice:
    def __init__(
        self,
        steering_bins: int,
        velocity_bins: int,
        velocity_min: float = 1.0,
        velocity_max: float = 8.0,
    ) -> None: ...
    @property
    def action_count(self) -> int: ...
    def normalized_action(self, action_index: int) -> torch.Tensor: ...
    def normalized_batch(self, action_indices: torch.Tensor) -> torch.Tensor: ...

class SearchBatchResult:
    action_probs: torch.Tensor
    metrics: dict[str, float]

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
    ) -> None: ...
    def search_batch(self, obs_batch: torch.Tensor) -> SearchBatchResult: ...

class SelfPlayEngine:
    def __init__(
        self,
        search: MuZeroSearchAdapter,
        track_map: TrackMap,
        obs_config: ObservationConfig,
        action_lattice: ActionLattice,
        discount: float = 0.997,
        sample_actions: bool = True,
        print_metrics: bool = False,
        waypoints_x: np.ndarray | list[float] = ...,
        waypoints_y: np.ndarray | list[float] = ...,
        cum_arc_lengths: np.ndarray | list[float] = ...,
        dynamics_params: F110Params = ...,
        car_length: float = 0.58,
        car_width: float = 0.31,
    ) -> None: ...
    def generate(
        self,
        rollout_steps: int,
        batch_size: int,
        initial_states: np.ndarray,
    ) -> SelfPlayResult: ...

class SelfPlayResult:
    trajectories: list[list[tuple[torch.Tensor, int, float, bool, torch.Tensor]]]
    metrics: dict[str, float]
