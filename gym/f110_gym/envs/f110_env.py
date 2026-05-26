# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Hongrui Zheng
"""

import time
from pathlib import Path

import gymnasium as gym

# others
import numpy as np

# base classes
from f110_gym.envs.base_classes import Integrator, Simulator
from gymnasium import spaces

# constants
DEFAULT_PARAMS = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I": 0.04712,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 20.0,
    "width": 0.31,
    "length": 0.58,
}

BUILTIN_MAPS = {
    "berlin": "berlin.yaml",
    "vegas": "vegas.yaml",
    "skirk": "skirk.yaml",
    "levine": "levine.yaml",
    "stata_basement": "stata_basement.yaml",
}

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


def _resolve_map_path(map_value):
    package_maps = Path(__file__).resolve().parent / "maps"
    candidate = Path(str(map_value))

    if candidate.suffix == ".yaml" and candidate.exists():
        return candidate

    if candidate.suffix and candidate.with_suffix(".yaml").exists():
        return candidate.with_suffix(".yaml")

    if str(map_value) in BUILTIN_MAPS:
        return package_maps / BUILTIN_MAPS[str(map_value)]

    if candidate.is_absolute():
        if candidate.suffix == ".yaml":
            return candidate
        return candidate.with_suffix(".yaml")

    builtin_candidate = package_maps / f"{map_value}.yaml"
    if builtin_candidate.exists():
        return builtin_candidate

    if candidate.suffix == ".yaml":
        return candidate
    return candidate.with_suffix(".yaml")


def _default_map_stem(map_path):
    return str(Path(map_path).with_suffix(""))


class F110Env(gym.Env):
    """
    Gymnasium environment for F1TENTH.

    Env should be initialized by calling gymnasium.make('f110-v0', **kwargs)

    Args:
        map: built-in map name or path to a map YAML file
        map_ext: image extension for custom map assets
        params: vehicle parameter dictionary
        num_agents: number of simulated vehicles
        timestep: physics time step
        ego_idx: index of the ego vehicle
        lidar_dist: vertical distance between LiDAR and backshaft
        seed: RNG seed used for scan noise
        render_mode: optional Gymnasium render mode
    """

    metadata = {"render_modes": ["human", "human_fast"], "render_fps": 200}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(
        self,
        map="vegas",
        map_ext=".png",
        params=None,
        num_agents=2,
        timestep=0.01,
        ego_idx=0,
        lidar_dist=0.0,
        integrator=Integrator.RK4,
        seed=12345,
        render_mode=None,
        **kwargs,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.seed = seed
        self.map_name = map
        self.map_path = str(_resolve_map_path(map))
        self.map_stem = _default_map_stem(self.map_path)
        self.map_ext = map_ext
        self.params = dict(DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)

        # simulation parameters
        self.num_agents = num_agents
        self.timestep = timestep
        self.ego_idx = ego_idx
        self.integrator = integrator
        self.lidar_dist = lidar_dist

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            ego_idx=self.ego_idx,
            integrator=self.integrator,
            lidar_dist=self.lidar_dist,
        )
        self.sim.set_map(self.map_path, self.map_ext)

        self._scan_length = self.sim.agents[0].num_beams
        self._scan_max_range = self.sim.agents[0].scan_simulator.max_range
        self.action_space = spaces.Box(
            low=np.tile(
                np.array(
                    [self.params["s_min"], self.params["v_min"]], dtype=np.float64
                ),
                (self.num_agents, 1),
            ),
            high=np.tile(
                np.array(
                    [self.params["s_max"], self.params["v_max"]], dtype=np.float64
                ),
                (self.num_agents, 1),
            ),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Discrete(self.num_agents),
                "scans": spaces.Box(
                    low=-1.0,
                    high=self._scan_max_range + 1.0,
                    shape=(self.num_agents, self._scan_length),
                    dtype=np.float64,
                ),
                "poses_x": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "poses_y": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "poses_theta": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "linear_vels_x": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "linear_vels_y": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "ang_vels_z": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "collisions": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_agents,), dtype=np.float64
                ),
                "lap_times": spaces.Box(
                    low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
                "lap_counts": spaces.Box(
                    low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64
                ),
            }
        )

        # stateful observations for rendering
        self.render_obs = None

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict["poses_x"]
        self.poses_y = obs_dict["poses_y"]
        self.poses_theta = obs_dict["poses_theta"]
        self.collisions = obs_dict["collisions"]

    def _coerce_poses(self, poses):
        if isinstance(poses, dict):
            poses = np.column_stack(
                (
                    np.asarray(poses["x"], dtype=np.float64),
                    np.asarray(poses["y"], dtype=np.float64),
                    np.asarray(poses["theta"], dtype=np.float64),
                )
            )
        else:
            poses = np.asarray(poses, dtype=np.float64)

        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("Reset poses must have shape (num_agents, 3).")

        return poses

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        obs = self.sim.step(action)

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        terminated, toggle_list = self._check_done()
        truncated = False
        info = {"checkpoint_done": toggle_list}

        obs["lap_times"] = self.lap_times
        obs["lap_counts"] = self.lap_counts
        F110Env.current_obs = obs

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
        }

        return obs, reward, terminated, truncated, info

    def reset(self, poses=None, *, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): initial observation of the environment
            info (dict): auxiliary information dictionary
        """
        super().reset(seed=seed)

        if poses is None and options is not None:
            poses = options.get("poses", options.get("initial_poses"))
        if poses is None:
            raise ValueError('reset requires a pose array or options["poses"].')

        poses = self._coerce_poses(poses)
        if seed is not None:
            self.seed = seed

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses, seed=self.seed)

        # get initial observations without advancing physics
        obs = self.sim.observe()
        obs["lap_times"] = self.lap_times
        obs["lap_counts"] = self.lap_counts
        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
        }

        self._update_state(obs)
        F110Env.current_obs = obs
        info = {"checkpoint_done": obs["lap_counts"] >= 4}

        return obs, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.map_path = str(map_path)
        self.map_stem = _default_map_stem(self.map_path)
        self.map_ext = map_ext
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1, *legacy_args, **kwargs):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters or legacy positional arguments
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        if isinstance(params, dict):
            self.params.update(params)
            self.sim.update_params(params, agent_idx=index)
            return

        # Legacy positional form:
        # update_params(mu, h, lr, C_Sf, C_Sr, I, m, executable_dir, ...)
        legacy_values = (params, index) + legacy_args
        if len(legacy_values) < 7:
            raise TypeError(
                "Legacy update_params requires at least 7 positional arguments."
            )

        mu, h, lr, C_Sf, C_Sr, I, m = legacy_values[:7]
        mapped = {
            "mu": mu,
            "h": h,
            "lr": lr,
            "C_Sf": C_Sf,
            "C_Sr": C_Sr,
            "I": I,
            "m": m,
        }
        self.params.update(mapped)
        self.sim.update_params(
            mapped, agent_idx=index if isinstance(index, int) else -1
        )

    def init_map(self, map_path, map_ext, *args, **kwargs):
        """
        Legacy alias for older helper scripts.
        """
        self.update_map(map_path, map_ext)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode=None):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default=None): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        render_mode = mode or self.render_mode or "human"
        assert render_mode in ["human", "human_fast"]

        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer

            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_stem, self.map_ext)

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if render_mode == "human":
            time.sleep(0.005)
        elif render_mode == "human_fast":
            pass
