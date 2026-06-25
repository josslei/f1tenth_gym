from __future__ import annotations

# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none

import queue
from pathlib import Path
from typing import Any, cast
import threading

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from planner.f110_self_play.backend import (
    ActionLattice,
    MuZeroSearchAdapter,
    SelfPlayEngine,
)
from .network import F110MuZeroNet
from .replay_buffer import MuZeroReplayBuffer


class PublishedModelClock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0

    def publish(self) -> int:
        with self._condition:
            self._version += 1
            self._condition.notify_all()
            return self._version


class _SelfPlayResultQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)

    def put(self, item: Any, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def get(self, stop_event: threading.Event) -> Any | None:
        while not stop_event.is_set():
            try:
                return self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
        return None


class LightningMuZero(pl.LightningModule):
    def __init__(
        self,
        model: F110MuZeroNet,
        replay_buffer: MuZeroReplayBuffer,
        training_config: dict[str, Any],
        replay_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.replay_buffer = replay_buffer
        self.training_config = training_config
        self.replay_config = replay_config

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.initial_training(obs)

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        (
            obs,
            actions,
            target_rewards,
            target_values,
            target_discounts,
            target_policies,
        ) = batch
        hidden, policy_logits, value = self.model.initial_training(obs)

        root_policy_loss = self._policy_loss(policy_logits, target_policies[:, 0])
        root_value_loss = F.mse_loss(value, target_values[:, 0])
        recurrent_policy_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]
        recurrent_value_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]
        reward_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]
        discount_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]

        for step in range(self.replay_config["unroll_steps"]):
            hidden, reward, value, discount, policy_logits = (
                self.model.recurrent_training(hidden, actions[:, step])
            )
            recurrent_policy_loss = recurrent_policy_loss + self._policy_loss(
                policy_logits, target_policies[:, step + 1]
            )
            recurrent_value_loss = recurrent_value_loss + F.mse_loss(
                value, target_values[:, step + 1]
            )
            reward_loss = reward_loss + F.mse_loss(reward, target_rewards[:, step])
            discount_loss = discount_loss + F.mse_loss(
                discount, target_discounts[:, step]
            )

        normalizer = float(self.replay_config["unroll_steps"] + 1)
        root_policy_loss = root_policy_loss / normalizer
        root_value_loss = root_value_loss / normalizer
        recurrent_policy_loss = recurrent_policy_loss / normalizer
        recurrent_value_loss = recurrent_value_loss / normalizer
        reward_loss = reward_loss / max(1.0, float(self.replay_config["unroll_steps"]))
        discount_loss = discount_loss / max(
            1.0, float(self.replay_config["unroll_steps"])
        )

        policy_loss = root_policy_loss + recurrent_policy_loss
        value_loss = root_value_loss + recurrent_value_loss
        representation_loss = root_policy_loss + root_value_loss
        prediction_loss = recurrent_policy_loss + recurrent_value_loss
        dynamics_loss = reward_loss + discount_loss

        loss = (
            self.training_config.get("policy_loss_weight", 1.0) * policy_loss
            + self.training_config.get("value_loss_weight", 1.0) * value_loss
            + self.training_config.get("reward_loss_weight", 1.0) * reward_loss
            + self.training_config.get("discount_loss_weight", 0.25) * discount_loss
        )

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/representation_loss",
            representation_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/prediction_loss", prediction_loss, on_step=False, on_epoch=True)
        self.log("train/dynamics_loss", dynamics_loss, on_step=False, on_epoch=True)
        self.log(
            "train/root_policy_loss", root_policy_loss, on_step=False, on_epoch=True
        )
        self.log("train/root_value_loss", root_value_loss, on_step=False, on_epoch=True)
        self.log(
            "train/recurrent_policy_loss",
            recurrent_policy_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/recurrent_value_loss",
            recurrent_value_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/policy_loss", policy_loss, on_step=False, on_epoch=True)
        self.log("train/value_loss", value_loss, on_step=False, on_epoch=True)
        self.log("train/reward_loss", reward_loss, on_step=False, on_epoch=True)
        self.log("train/discount_loss", discount_loss, on_step=False, on_epoch=True)
        self.log(
            "replay/size", float(len(self.replay_buffer)), on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.training_config["max_grad_norm"]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.replay_buffer,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            num_workers=self.replay_config.get("num_workers", 0),
            pin_memory=True,
            persistent_workers=self.replay_config.get("num_workers", 0) > 0,
        )

    @staticmethod
    def _policy_loss(policy_logits: Tensor, target_policy: Tensor) -> Tensor:
        log_probs = F.log_softmax(policy_logits, dim=-1)
        return -(target_policy * log_probs).sum(dim=-1).mean()


class TorchScriptExportCallback(pl.Callback):
    def __init__(self, output_dir: str, model_clock: PublishedModelClock) -> None:
        self.output_dir = Path(output_dir)
        self.model_clock = model_clock

    def _export(self, pl_module: pl.LightningModule) -> None:
        module = cast(LightningMuZero, pl_module)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "current_model.pt"
        scripted = torch.jit.script(module.model.eval())
        scripted.save(str(path))
        module.model.train()
        self.model_clock.publish()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._export(pl_module)


class SelfPlayCallback(pl.Callback):
    def __init__(
        self,
        model_clock: PublishedModelClock,
        hidden_size: int,
        track_map: Any,
        observation_config: Any,
        action_lattice: ActionLattice,
        waypoints: Any,
        cum_arc_lengths: Any,
        dynamics_params: Any,
        initial_states: Any,
        car_length: float,
        car_width: float,
        reward_config: dict[str, Any],
        rollout_steps: int,
        num_iters: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        temperature: float,
        search_print_metrics: bool,
        self_play_print_metrics: bool,
        discount: float,
        device: Any,
        model_path: str,
    ) -> None:
        self.track_map = track_map
        self.observation_config = observation_config
        self.action_lattice = action_lattice
        self.waypoints = waypoints
        self.cum_arc_lengths = cum_arc_lengths
        self.dynamics_params = dynamics_params
        self.initial_states = initial_states
        self.car_length = car_length
        self.car_width = car_width
        self.reward_config = reward_config
        self.model_clock = model_clock
        self.hidden_size = hidden_size
        self.rollout_steps = rollout_steps
        self.num_iters = num_iters
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.search_print_metrics = search_print_metrics
        self.self_play_print_metrics = self_play_print_metrics
        self.discount = discount
        self.device = device
        self.model_path = Path(model_path)
        self.result_queue = _SelfPlayResultQueue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def _generate(self) -> Any:
        with self.model_clock._condition:
            search = MuZeroSearchAdapter(
                str(self.model_path),
                self.num_iters,
                self.temperature,
                self.c_puct,
                self.dirichlet_alpha,
                self.dirichlet_epsilon,
                int(self.initial_states.shape[0]),
                self.action_lattice.action_count,
                self.hidden_size,
                0,
                str(self.device).split(":", maxsplit=1)[0],
                self.search_print_metrics,
            )
        return SelfPlayEngine(
            search,
            self.track_map,
            self.observation_config,
            self.action_lattice,
            self.discount,
            True,
            self.self_play_print_metrics,
            self.waypoints[:, 0],
            self.waypoints[:, 1],
            self.cum_arc_lengths,
            self.dynamics_params,
            self.car_length,
            self.car_width,
            self.reward_config["q_s_progress"],
            self.reward_config["q_s_alpha"],
            self.reward_config["q_s_smooth"],
            self.reward_config["terminal_penalty"],
            self.reward_config["alpha_th"],
            self.reward_config["slip_terminal_penalty"],
            self.reward_config["q_offtrack_grad"],
        ).generate(
            self.rollout_steps,
            int(self.initial_states.shape[0]),
            self.initial_states,
        )

    def _worker_loop(self) -> None:
        last_version = 0
        while not self.stop_event.is_set():
            with self.model_clock._condition:
                self.model_clock._condition.wait_for(
                    lambda: self.stop_event.is_set()
                    or self.model_clock._version > last_version
                )
                if self.stop_event.is_set():
                    return
                model_version = self.model_clock._version
            result = self._generate()
            if self.stop_event.is_set():
                return
            self.result_queue.put((model_version, result), self.stop_event)
            last_version = model_version

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.stop_event.set()
        with self.model_clock._condition:
            self.model_clock._condition.notify_all()
        if self.worker is not None:
            self.worker.join()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        module = cast(LightningMuZero, pl_module)
        pending = self.result_queue.get(self.stop_event)
        if pending is None:
            return
        _model_version, result = pending
        for trajectory in result.trajectories:
            module.replay_buffer.push(trajectory)
        for key, value in result.metrics.items():
            module.log(key, value, on_step=False, on_epoch=True)


__all__ = ["LightningMuZero", "SelfPlayCallback", "TorchScriptExportCallback"]
