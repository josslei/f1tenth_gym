from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.ppo import (
    PPOActionConfig,
    PPOObservationConfig,
    build_ppo_observation,
    make_policy,
    observation_dim,
    scale_ppo_action,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "env": {"map": "vegas", "seed": 12345, "max_episode_steps": 1000},
    "observation": {
        "scan_size": 108,
        "scan_max_m": 30.0,
        "include_ego_state": True,
        "speed_scale": 8.0,
        "yaw_rate_scale": 10.0,
    },
    "action": {"velocity_min": 0.0, "velocity_max": 8.0},
    "reward": {
        "alive_reward": 0.01,
        "speed_weight": 0.05,
        "steering_penalty_weight": 0.01,
        "collision_penalty": 5.0,
        "lap_bonus": 10.0,
        "target_speed": 5.0,
    },
    "training": {
        "epochs": 10,
        "total_timesteps": 4096,
        "rollout_steps": 128,
        "ppo_epochs": 4,
        "hidden_size": 128,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    },
    "output": {"root": "outputs/rl/ppo_vegas", "metrics_file": "metrics.jsonl"},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "ppo" / "default.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact Lightning PPO policy")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with config_path.expanduser().open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    return _deep_update(DEFAULT_CONFIG, loaded)


def _resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    env_cfg = resolved.setdefault("env", {})
    training_cfg = resolved.setdefault("training", {})
    output_cfg = resolved.setdefault("output", {})

    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.total_timesteps is not None:
        training_cfg["total_timesteps"] = int(args.total_timesteps)
    if args.rollout_steps is not None:
        training_cfg["rollout_steps"] = int(args.rollout_steps)
    if args.map is not None:
        env_cfg["map"] = args.map
    if args.seed is not None:
        env_cfg["seed"] = int(args.seed)
    if args.max_episode_steps is not None:
        env_cfg["max_episode_steps"] = int(args.max_episode_steps)
    if args.output_dir is not None:
        output_cfg["root"] = str(args.output_dir)
    output_cfg.setdefault("metrics_file", "metrics.jsonl")
    return resolved


def _load_lightning() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset

        try:
            import lightning.pytorch as pl
        except ImportError:
            import pytorch_lightning as pl
    except ImportError as exc:
        raise SystemExit(
            "PyTorch Lightning is required for PPO training. Install it with: "
            'pip install -e ".[rl]"'
        ) from exc
    return (
        torch,
        DataLoader,
        Dataset,
        pl.LightningModule,
        pl.Trainer,
        pl.callbacks.ModelCheckpoint,
        pl.loggers.CSVLogger,
        pl.loggers.TensorBoardLogger,
    )


def _map_center_pose(env: gym.Env) -> np.ndarray:
    unwrapped: Any = env.unwrapped
    scan_sim = unwrapped.sim.agents[0].scan_simulator
    origin_x, origin_y, _ = scan_sim.origin
    center_x = origin_x + scan_sim.map_width * scan_sim.map_resolution / 2.0
    center_y = origin_y + scan_sim.map_height * scan_sim.map_resolution / 2.0
    return np.array([[center_x, center_y, 0.0]], dtype=np.float64)


def _ego(obs: dict[str, Any]) -> int:
    return int(np.asarray(obs.get("ego_idx", 0)).item())


def _shaped_reward(
    obs: dict[str, Any],
    raw_reward: float,
    normalized_action: np.ndarray,
    previous_lap_count: int,
    reward_cfg: dict[str, Any],
    info: dict[str, Any],
) -> tuple[float, int]:
    ego = _ego(obs)
    collision = float(np.asarray(obs["collisions"], dtype=np.float64)[ego])
    speed = float(np.asarray(obs["linear_vels_x"], dtype=np.float64)[ego])
    lap_count = int(np.asarray(obs["lap_counts"], dtype=np.int64)[ego])
    checkpoints = np.asarray(info.get("checkpoint_done", [False]), dtype=bool)
    lap_done = bool(checkpoints[ego]) or lap_count > previous_lap_count
    target_speed = float(reward_cfg.get("target_speed", 5.0))
    speed_norm = float(np.clip(speed / target_speed, -1.0, 1.0))
    reward = (
        float(raw_reward)
        + float(reward_cfg.get("alive_reward", 0.01))
        + float(reward_cfg.get("speed_weight", 0.05)) * speed_norm
        - float(reward_cfg.get("steering_penalty_weight", 0.01))
        * abs(float(normalized_action[0]))
        - float(reward_cfg.get("collision_penalty", 5.0)) * collision
        + float(reward_cfg.get("lap_bonus", 10.0)) * float(lap_done)
    )
    return reward, lap_count


def _make_update_dataset(dataset_cls: type[Any], updates: int) -> Any:
    class UpdateDataset(dataset_cls):
        def __len__(self) -> int:
            return updates

        def __getitem__(self, index: int) -> int:
            return index

    return UpdateDataset()


def _make_module(
    lightning_module_cls: type[Any],
    torch: Any,
    env: gym.Env,
    config: dict[str, Any],
    output_dir: Path,
) -> Any:
    observation_config = PPOObservationConfig(**config.get("observation", {}))
    action_config = PPOActionConfig(**config.get("action", {}))
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    env_cfg = config.get("env", {})
    obs_dim = observation_dim(observation_config)
    hidden_size = int(training_cfg.get("hidden_size", 128))
    metrics_path = output_dir / config.get("output", {}).get(
        "metrics_file", "metrics.jsonl"
    )
    updates_path = output_dir / "updates.jsonl"

    class LightningPPO(lightning_module_cls):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(
                {
                    "obs_dim": obs_dim,
                    "action_dim": 2,
                    "hidden_size": hidden_size,
                    "observation_config": observation_config.__dict__,
                    "action_config": action_config.__dict__,
                }
            )
            self.policy = make_policy(obs_dim=obs_dim, hidden_size=hidden_size)
            self.automatic_optimization = False
            self.env = env
            self.obs: dict[str, Any] | None = None
            self.previous_lap_count = 0
            self.episode_return = 0.0
            self.episodes_seen = 0
            self.max_episode_steps = int(env_cfg.get("max_episode_steps", 1000))
            self.episode_steps = 0
            self.completed_episode_returns: list[float] = []

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.policy.parameters(),
                lr=float(training_cfg.get("learning_rate", 3e-4)),
            )

        def _reset_env(self) -> np.ndarray:
            obs, _info = self.env.reset(options={"poses": _map_center_pose(self.env)})
            self.obs = obs
            self.previous_lap_count = int(
                np.asarray(obs["lap_counts"], dtype=np.int64)[_ego(obs)]
            )
            self.episode_return = 0.0
            self.episode_steps = 0
            return build_ppo_observation(obs, observation_config)

        def _collect_rollout(self) -> dict[str, Any]:
            if self.obs is None:
                current_obs = self._reset_env()
            else:
                current_obs = build_ppo_observation(self.obs, observation_config)

            rollout_steps = int(training_cfg.get("rollout_steps", 128))
            obs_values: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            log_probs: list[float] = []
            rewards: list[float] = []
            raw_rewards: list[float] = []
            dones: list[float] = []
            values: list[float] = []

            for _step in range(rollout_steps):
                obs_tensor = torch.as_tensor(
                    current_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, value_tensor = self.policy.act(
                        obs_tensor
                    )
                normalized_action = action_tensor.squeeze(0).cpu().numpy()
                env_action = scale_ppo_action(
                    normalized_action, self.env, action_config
                )
                next_obs, raw_reward, terminated, truncated, info = self.env.step(
                    env_action
                )
                shaped_reward, self.previous_lap_count = _shaped_reward(
                    next_obs,
                    float(raw_reward),
                    normalized_action,
                    self.previous_lap_count,
                    reward_cfg,
                    info,
                )
                self.episode_return += shaped_reward
                self.episode_steps += 1
                timeout = self.episode_steps >= self.max_episode_steps
                done = bool(terminated or truncated or timeout)

                obs_values.append(current_obs)
                actions.append(normalized_action.astype(np.float32))
                log_probs.append(float(log_prob_tensor.item()))
                rewards.append(float(shaped_reward))
                raw_rewards.append(float(raw_reward))
                dones.append(float(done))
                values.append(float(value_tensor.item()))

                if done:
                    self.episodes_seen += 1
                    self.completed_episode_returns.append(float(self.episode_return))
                    self._write_metrics(self.episode_return)
                    current_obs = self._reset_env()
                else:
                    self.obs = next_obs
                    current_obs = build_ppo_observation(next_obs, observation_config)

            next_value = 0.0
            if self.obs is not None:
                obs_tensor = torch.as_tensor(
                    current_obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    _mean, value_tensor = self.policy(obs_tensor)
                next_value = float(value_tensor.item())

            advantages = self._gae(rewards, dones, values, next_value)
            returns = advantages + np.asarray(values, dtype=np.float32)
            reward_array = np.asarray(rewards, dtype=np.float32)
            raw_reward_array = np.asarray(raw_rewards, dtype=np.float32)
            action_array = np.asarray(actions, dtype=np.float32)
            value_array = np.asarray(values, dtype=np.float32)
            return {
                "obs": torch.as_tensor(np.asarray(obs_values), dtype=torch.float32),
                "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32),
                "old_log_probs": torch.as_tensor(log_probs, dtype=torch.float32),
                "advantages": torch.as_tensor(advantages, dtype=torch.float32),
                "returns": torch.as_tensor(returns, dtype=torch.float32),
                "stats": {
                    "reward_sum": float(reward_array.sum()),
                    "reward_mean": float(reward_array.mean()),
                    "reward_min": float(reward_array.min()),
                    "reward_max": float(reward_array.max()),
                    "raw_reward_mean": float(raw_reward_array.mean()),
                    "done_count": int(np.asarray(dones, dtype=np.float32).sum()),
                    "action_steering_mean": float(action_array[:, 0].mean()),
                    "action_steering_std": float(action_array[:, 0].std()),
                    "action_velocity_mean": float(action_array[:, 1].mean()),
                    "action_velocity_std": float(action_array[:, 1].std()),
                    "value_mean": float(value_array.mean()),
                    "value_std": float(value_array.std()),
                    "episodes_seen": int(self.episodes_seen),
                    "last_episode_return": (
                        self.completed_episode_returns[-1]
                        if self.completed_episode_returns
                        else None
                    ),
                    "mean_episode_return": (
                        float(np.mean(self.completed_episode_returns[-10:]))
                        if self.completed_episode_returns
                        else None
                    ),
                },
            }

        def _gae(
            self,
            rewards: list[float],
            dones: list[float],
            values: list[float],
            next_value: float,
        ) -> np.ndarray:
            gamma = float(training_cfg.get("gamma", 0.99))
            gae_lambda = float(training_cfg.get("gae_lambda", 0.95))
            advantages = np.zeros(len(rewards), dtype=np.float32)
            gae = 0.0
            for step in reversed(range(len(rewards))):
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value if step == len(rewards) - 1 else values[step + 1]
                delta = (
                    rewards[step] + gamma * next_val * next_non_terminal - values[step]
                )
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                advantages[step] = gae
            return advantages

        def _write_metrics(self, episode_return: float) -> None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "episodes_seen": int(self.episodes_seen),
                "episode_return": float(episode_return),
                "global_step": int(self.global_step),
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(payload) + "\n")

        def _write_update_metrics(self, payload: dict[str, Any]) -> None:
            updates_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                key: value
                for key, value in payload.items()
                if value is None or isinstance(value, int | float | str | bool)
            }
            with updates_path.open("a", encoding="utf-8") as updates_file:
                updates_file.write(json.dumps(serializable) + "\n")

        def training_step(self, batch, batch_idx):
            optimizer = self.optimizers()
            batch_data = self._collect_rollout()
            obs = batch_data["obs"].to(self.device)
            actions = batch_data["actions"].to(self.device)
            old_log_probs = batch_data["old_log_probs"].to(self.device)
            advantages = batch_data["advantages"].to(self.device)
            returns = batch_data["returns"].to(self.device)
            rollout_stats = dict(batch_data["stats"])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            clip_range = float(training_cfg.get("clip_range", 0.2))
            value_coef = float(training_cfg.get("value_coef", 0.5))
            entropy_coef = float(training_cfg.get("entropy_coef", 0.01))
            ppo_epochs = max(1, int(training_cfg.get("ppo_epochs", 4)))
            log_probs = old_log_probs
            ratio = torch.ones_like(old_log_probs)
            policy_loss = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            loss = torch.tensor(0.0, device=self.device)

            for _epoch in range(ppo_epochs):
                log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = (
                    torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
                )
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - log_probs).mean()
                clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
            update_metrics = {
                "train/loss_total": float(loss.detach().cpu()),
                "train/loss_policy": float(policy_loss.detach().cpu()),
                "train/loss_value": float(value_loss.detach().cpu()),
                "train/entropy": float(entropy_loss.detach().cpu()),
                "train/approx_kl": float(approx_kl.detach().cpu()),
                "train/clip_fraction": float(clip_frac.detach().cpu()),
                "train/advantage_mean": float(advantages.mean().detach().cpu()),
                "train/advantage_std": float(advantages.std().detach().cpu()),
                "train/return_mean": float(returns.mean().detach().cpu()),
                "train/return_std": float(returns.std().detach().cpu()),
                "reward/rollout_sum": rollout_stats["reward_sum"],
                "reward/rollout_mean": rollout_stats["reward_mean"],
                "reward/rollout_min": rollout_stats["reward_min"],
                "reward/rollout_max": rollout_stats["reward_max"],
                "reward/raw_mean": rollout_stats["raw_reward_mean"],
                "reward/last_episode_return": rollout_stats["last_episode_return"],
                "reward/mean_episode_return_10": rollout_stats["mean_episode_return"],
                "env/done_count": rollout_stats["done_count"],
                "env/episodes_seen": rollout_stats["episodes_seen"],
                "action/steering_mean": rollout_stats["action_steering_mean"],
                "action/steering_std": rollout_stats["action_steering_std"],
                "action/velocity_mean": rollout_stats["action_velocity_mean"],
                "action/velocity_std": rollout_stats["action_velocity_std"],
                "value/prediction_mean": rollout_stats["value_mean"],
                "value/prediction_std": rollout_stats["value_std"],
            }
            loggable_metrics = {
                key: value for key, value in update_metrics.items() if value is not None
            }
            self.log_dict(
                loggable_metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            update_metrics["global_step"] = int(self.global_step)
            self._write_update_metrics(update_metrics)

    return LightningPPO()


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")


def main() -> int:
    args = _parse_args()
    (
        torch,
        DataLoader,
        Dataset,
        LightningModule,
        Trainer,
        ModelCheckpoint,
        CSVLogger,
        TensorBoardLogger,
    ) = _load_lightning()
    config = _resolve_config(_load_config(args.config), args)
    seed = config.get("env", {}).get("seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    output_dir = Path(config["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / config["output"].get("metrics_file", "metrics.jsonl")
    if metrics_path.exists():
        metrics_path.unlink()
    updates_path = output_dir / "updates.jsonl"
    if updates_path.exists():
        updates_path.unlink()

    env = gym.make("f110-v0", map=config["env"].get("map", "vegas"), num_agents=1)
    total_timesteps = int(config["training"].get("total_timesteps", 4096))
    rollout_steps = int(config["training"].get("rollout_steps", 128))
    epochs = max(1, int(config["training"].get("epochs", 10)))
    updates = max(1, math.ceil(total_timesteps / rollout_steps))
    module = _make_module(LightningModule, torch, env, config, output_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="ppo-{step}",
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=epochs,
        limit_train_batches=updates,
        callbacks=[checkpoint_callback],
        logger=[
            CSVLogger(save_dir=str(output_dir), name="lightning_csv"),
            TensorBoardLogger(save_dir=str(output_dir), name="tensorboard"),
        ],
        enable_progress_bar=not args.no_progress_bar,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    try:
        dataset = _make_update_dataset(Dataset, updates)
        trainer.fit(module, train_dataloaders=DataLoader(dataset, batch_size=1))
    finally:
        env.close()

    final_model_path = output_dir / "final_model.pt"
    torch.save(
        {
            "policy_state_dict": module.policy.state_dict(),
            "obs_dim": observation_dim(
                PPOObservationConfig(**config.get("observation", {}))
            ),
            "action_dim": 2,
            "hidden_size": int(config["training"].get("hidden_size", 128)),
            "config": config,
        },
        final_model_path,
    )
    _write_summary(
        output_dir / "run_summary.json",
        {
            "config_path": str(args.config),
            "output_dir": str(output_dir),
            "total_timesteps": total_timesteps,
            "rollout_steps": rollout_steps,
            "epochs": epochs,
            "updates": updates,
            "estimated_env_steps": epochs * updates * rollout_steps,
            "final_model_path": str(final_model_path),
            "metrics_path": str(metrics_path),
            "updates_path": str(updates_path),
            "csv_log_dir": str(output_dir / "lightning_csv"),
            "tensorboard_log_dir": str(output_dir / "tensorboard"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
