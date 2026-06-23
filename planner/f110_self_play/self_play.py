from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .backend import ActionLattice, MuZeroSearchAdapter


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    done: bool
    root_policy: torch.Tensor


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)


@dataclass
class StepBatchResult:
    obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    lap_count: torch.Tensor
    lap_time: torch.Tensor
    collision: torch.Tensor
    reset_obs: torch.Tensor


@dataclass
class SelfPlayResult:
    trajectories: list[Trajectory] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class SelfPlayEngine:
    def __init__(
        self,
        search: MuZeroSearchAdapter,
        env: Any,
        action_lattice: ActionLattice,
        discount: float = 0.997,
        sample_actions: bool = True,
        print_metrics: bool = False,
    ) -> None:
        self.search = search
        self.env = env
        self.action_lattice = action_lattice
        self.discount = discount
        self.sample_actions = sample_actions
        self.print_metrics = print_metrics
        self._rng = np.random.default_rng()

    def generate(self, rollout_steps: int) -> SelfPlayResult:
        obs = self.env.reset_batch()
        batch_size = obs.size(0)

        active: list[Trajectory] = [Trajectory() for _ in range(batch_size)]
        completed: list[Trajectory] = []

        ep_return = [0.0] * batch_size
        ep_discounted = [0.0] * batch_size
        ep_length = [0] * batch_size
        ep_lap_count = [0] * batch_size
        ep_lap_time = [0.0] * batch_size
        ep_collision = [False] * batch_size
        ep_completed = [False] * batch_size
        ep_truncated = [False] * batch_size

        search_metrics_sum: dict[str, float] = {}
        search_steps = 0

        for _ in range(rollout_steps):
            prev_obs = obs
            result = self.search.search_batch(obs)
            for k, v in result.metrics.items():
                search_metrics_sum[k] = search_metrics_sum.get(k, 0.0) + v
            search_steps += 1

            action_probs = result.action_probs.cpu().contiguous()
            action_indices = self._sample_action_batch(action_probs)
            normalized_actions = self.action_lattice.normalized_batch(action_indices)
            step_result = self.env.step_batch(normalized_actions)

            reward = step_result.reward.cpu().contiguous()
            terminated = step_result.terminated.cpu().contiguous()
            truncated = step_result.truncated.cpu().contiguous()
            lap_count = step_result.lap_count.cpu().contiguous()
            lap_time = step_result.lap_time.cpu().contiguous()
            collision = step_result.collision.cpu().contiguous()

            for b in range(batch_size):
                obs_row = prev_obs.select(0, b).clone()
                policy_row = action_probs.select(0, b).clone()
                action = int(action_indices[b].item())
                r = float(reward[b].item())
                done = bool(terminated[b].item()) or bool(truncated[b].item())

                active[b].transitions.append(
                    Transition(
                        obs=obs_row,
                        action=action,
                        reward=r,
                        done=done,
                        root_policy=policy_row,
                    )
                )

                ep_return[b] += r
                ep_discounted[b] += (self.discount ** ep_length[b]) * r
                ep_length[b] += 1
                ep_lap_count[b] = int(lap_count[b].item())
                ep_lap_time[b] = float(lap_time[b].item())
                ep_collision[b] = ep_collision[b] or bool(collision[b].item())
                ep_completed[b] = ep_completed[b] or (
                    done and not bool(truncated[b].item())
                )
                ep_truncated[b] = ep_truncated[b] or bool(truncated[b].item())

                if done:
                    completed.append(active[b])
                    active[b] = Trajectory()
                    ep_return[b] = 0.0
                    ep_discounted[b] = 0.0
                    ep_length[b] = 0
                    ep_lap_count[b] = 0
                    ep_lap_time[b] = 0.0
                    ep_collision[b] = False
                    ep_completed[b] = False
                    ep_truncated[b] = False

            obs = step_result.obs
            reset_obs_cpu = step_result.reset_obs.cpu().contiguous()
            for b in range(batch_size):
                if bool(terminated[b].item()) or bool(truncated[b].item()):
                    obs[b] = reset_obs_cpu[b]

        for traj in active:
            if traj.transitions:
                completed.append(traj)

        metrics = self._build_metrics(
            ep_return,
            ep_discounted,
            ep_length,
            ep_lap_count,
            ep_lap_time,
            ep_collision,
            ep_completed,
            ep_truncated,
            search_metrics_sum,
            search_steps,
            batch_size,
            len(completed),
        )

        if self.print_metrics:
            self._print_metrics_summary(metrics)

        return SelfPlayResult(trajectories=completed, metrics=metrics)

    def _sample_action_batch(self, action_probs: torch.Tensor) -> torch.Tensor:
        batch_size = action_probs.size(0)
        action_count = action_probs.size(1)
        actions = torch.empty(batch_size, dtype=torch.long)
        probs_np = action_probs.numpy()
        for b in range(batch_size):
            row = probs_np[b]
            s = float(row.sum())
            if not self.sample_actions or s <= 0.0:
                actions[b] = int(np.argmax(row))
            else:
                actions[b] = int(self._rng.choice(action_count, p=row / s))
        return actions

    @staticmethod
    def _build_metrics(
        ep_return,
        ep_discounted,
        ep_length,
        ep_lap_count,
        ep_lap_time,
        ep_collision,
        ep_completed,
        ep_truncated,
        search_metrics_sum,
        search_steps,
        batch_size,
        num_trajectories,
    ) -> dict[str, float]:
        m: dict[str, float] = dict(search_metrics_sum)
        total_transitions = search_steps * batch_size
        m["self_play/search_steps"] = float(search_steps)
        m["self_play/transitions"] = float(total_transitions)
        m["self_play/trajectories"] = float(num_trajectories)
        m["self_play/samples"] = float(total_transitions)

        count = sum(1 for v in ep_length if v > 0)
        if count > 0:
            m["episode/count"] = float(count)
            m["episode/return_sum"] = sum(ep_return)
            m["episode/discounted_return_sum"] = sum(ep_discounted)
            m["episode/length_sum"] = float(sum(ep_length))
            m["episode/lap_count_sum"] = float(sum(ep_lap_count))
            m["episode/lap_time_sum"] = sum(ep_lap_time)
            m["episode/collision_count"] = float(sum(ep_collision))
            m["episode/completed_count"] = float(sum(ep_completed))
            m["episode/truncated_count"] = float(sum(ep_truncated))

            c = float(count)
            m["episode/return_mean"] = m["episode/return_sum"] / c
            m["episode/discounted_return_mean"] = m["episode/discounted_return_sum"] / c
            m["episode/length_mean"] = m["episode/length_sum"] / c
            m["episode/lap_count_mean"] = m["episode/lap_count_sum"] / c
            m["episode/lap_time_mean"] = m["episode/lap_time_sum"] / c
            m["episode/collision_rate"] = m["episode/collision_count"] / c
            m["episode/completion_rate"] = m["episode/completed_count"] / c
            m["episode/truncation_rate"] = m["episode/truncated_count"] / c

        if search_steps > 0:
            s = float(search_steps)
            for k in list(m.keys()):
                if (
                    k.startswith("search/")
                    or k.startswith("inference/")
                    or k.startswith("tree/")
                    or k.startswith("throughput/")
                ):
                    m[k] /= s

        return m

    def _print_metrics_summary(self, m: dict[str, float]) -> None:
        def v(key: str) -> float:
            return m.get(key, 0.0)

        print("---------------- Self-Play Metrics ----------------", flush=True)
        print("[Rollout]", flush=True)
        print(
            f"  search steps: {v('self_play/search_steps')}"
            f" | transitions: {v('self_play/transitions')}"
            f" | samples: {v('self_play/samples')}",
            flush=True,
        )
        print(f"  trajectories: {v('self_play/trajectories')}", flush=True)
        print("[Episode]", flush=True)
        print(
            f"  count: {v('episode/count')}"
            f" | length mean: {v('episode/length_mean')}",
            flush=True,
        )
        print(
            f"  return mean: {v('episode/return_mean')}"
            f" | discounted: {v('episode/discounted_return_mean')}",
            flush=True,
        )
        print(
            f"  completion: {v('episode/completion_rate')}"
            f" | collision: {v('episode/collision_rate')}"
            f" | truncation: {v('episode/truncation_rate')}",
            flush=True,
        )
        print("[Search Average]", flush=True)
        print(
            f"  total:   {v('search/total_time_us') / 1000.0:.4f} ms",
            flush=True,
        )
        print(
            f"  select:  {v('search/selection_time_us') / 1000.0:.4f}"
            f" ms | expand: {v('search/expand_time_us') / 1000.0:.4f}"
            f" ms | backup: {v('search/backup_time_us') / 1000.0:.4f} ms",
            flush=True,
        )
        print(
            f"  policy:  {v('search/root_policy_time_us') / 1000.0:.4f} ms",
            flush=True,
        )
        print("[Inference Average]", flush=True)
        print(
            f"  initial:  {v('inference/initial_time_us') / 1000.0:.4f}"
            f" ms | recurrent: {v('inference/recurrent_time_us') / 1000.0:.4f}"
            f" ms | copy: {v('inference/payload_copy_time_us') / 1000.0:.4f} ms",
            flush=True,
        )
        print("[Tree Average]", flush=True)
        print(
            f"  nodes avg/min/max: {v('tree/nodes_allocated_avg')}"
            f" / {v('tree/nodes_allocated_min')}"
            f" / {v('tree/nodes_allocated_max')}",
            flush=True,
        )
        print(
            f"  depth avg/min/max: {v('tree/search_depth_avg')}"
            f" / {v('tree/search_depth_min')}"
            f" / {v('tree/search_depth_max')}",
            flush=True,
        )
        print(
            f"  root visits avg/min/max: {v('tree/root_visit_count_avg')}"
            f" / {v('tree/root_visit_count_min')}"
            f" / {v('tree/root_visit_count_max')}",
            flush=True,
        )
        print(
            f"  simulations/sec: {v('throughput/simulations_per_second')}",
            flush=True,
        )
        print("---------------------------------------------------", flush=True)
