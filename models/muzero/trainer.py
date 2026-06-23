from __future__ import annotations

# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none

from pathlib import Path
from typing import Any, cast

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

        policy_loss = self._policy_loss(policy_logits, target_policies[:, 0])
        value_loss = F.mse_loss(value, target_values[:, 0])
        reward_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]
        discount_loss = torch.zeros((), device=obs.device)  # type: ignore[attr-defined]

        for step in range(self.replay_config["unroll_steps"]):
            hidden, reward, value, discount, policy_logits = (
                self.model.recurrent_training(hidden, actions[:, step])
            )
            policy_loss = policy_loss + self._policy_loss(
                policy_logits, target_policies[:, step + 1]
            )
            value_loss = value_loss + F.mse_loss(value, target_values[:, step + 1])
            reward_loss = reward_loss + F.mse_loss(reward, target_rewards[:, step])
            discount_loss = discount_loss + F.mse_loss(
                discount, target_discounts[:, step + 1]
            )

        normalizer = float(self.replay_config["unroll_steps"] + 1)
        policy_loss = policy_loss / normalizer
        value_loss = value_loss / normalizer
        reward_loss = reward_loss / max(1.0, float(self.replay_config["unroll_steps"]))
        discount_loss = discount_loss / normalizer

        loss = (
            self.training_config.get("policy_loss_weight", 1.0) * policy_loss
            + self.training_config.get("value_loss_weight", 1.0) * value_loss
            + self.training_config.get("reward_loss_weight", 1.0) * reward_loss
            + self.training_config.get("discount_loss_weight", 0.25) * discount_loss
        )

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        module = cast(LightningMuZero, pl_module)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "current_model.pt"
        scripted = torch.jit.script(module.model.eval())
        scripted.save(str(path))
        module.model.train()


class SelfPlayCallback(pl.Callback):
    def __init__(
        self,
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
        self.rollout_steps = rollout_steps
        self.num_iters = num_iters
        self.c_puct = c_puct
        self.temperature = temperature
        self.search_print_metrics = search_print_metrics
        self.self_play_print_metrics = self_play_print_metrics
        self.discount = discount
        self.device = device
        self.model_path = Path(model_path)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        module = cast(LightningMuZero, pl_module)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.script(module.model.eval()).to(self.device)
        scripted.save(str(self.model_path))
        module.model.train()

        search = MuZeroSearchAdapter(
            str(self.model_path),
            self.num_iters,
            self.temperature,
            self.c_puct,
            int(self.initial_states.shape[0]),
            self.action_lattice.action_count,
            module.model.hidden_size,
            0,
            str(self.device).split(":", maxsplit=1)[0],
            self.search_print_metrics,
        )
        result = SelfPlayEngine(
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
            self.reward_config["speed_reward_weight"],
            self.reward_config["progress_weight"],
            self.reward_config["steer_smoothness_weight"],
            self.reward_config["collision_penalty"],
            self.reward_config["spin_threshold"],
        ).generate(
            self.rollout_steps,
            int(self.initial_states.shape[0]),
            self.initial_states,
        )
        module.model.train()
        for trajectory in result.trajectories:
            module.replay_buffer.push(trajectory)
        for key, value in result.metrics.items():
            module.log(key, value, on_step=False, on_epoch=True)


__all__ = ["LightningMuZero", "SelfPlayCallback", "TorchScriptExportCallback"]
