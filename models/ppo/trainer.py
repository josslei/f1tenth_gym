"""PPO Lightning module and closely related algorithmic helpers."""

from __future__ import annotations

from typing import Any, Tuple, cast

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import PPOConfig
from models.policies import Policy

# Public API
__all__ = [
    "compute_gae",
    "ppo_loss",
    "LightningPPO",
]


# ═══════════════════════════════════════════════════════════════════════════════
# LightningPPO – PyTorch Lightning training harness
# ═══════════════════════════════════════════════════════════════════════════════


class LightningPPO(pl.LightningModule):
    """PyTorch Lightning module for PPO training.

    Encapsulates the update part of Algorithm 2 after the trajectory dataset
    ``D`` has already been sampled externally: build minibatches ``B`` from
    ``I = {(j, t)}``, repeat for configured update epochs, and optimize
    ``L_PPO = L_CLIP - c1 L_V + c2 L_H``.

    Fixed ``training_step`` batch format:

    ``(s, a, log_p_cur, A_hat, R_hat, episode_return, lap_number, lap_time, completed_episodes)``

    Common names: ``s`` = observations/states, ``a`` = actions,
    ``log_p_cur`` = old log probabilities, ``A_hat`` = advantages,
    ``R_hat`` = returns. The final four scalars are rollout-level
    training metrics logged through Lightning.

    Parameters
    ----------
    policy : Policy
        An actor-critic network exposing ``evaluate_actions(obs, actions)``
        that returns ``(log_probs, entropy, values)``.
    config : PPOConfig
        Structured config loaded from ``configs/ppo/*.yaml``. This module only
        reads ``config.training``.
    """

    def __init__(
        self,
        policy: Policy,
        config: PPOConfig,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.policy = policy
        self.config = config

    def forward(self, obs: Tensor) -> Any:  # noqa: D102
        return self.policy(obs)

    def training_step(  # noqa: D102
        self,
        batch: Tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ],
        batch_idx: int,
    ) -> Tensor:
        (
            s,
            a,
            log_p_cur,
            A_hat,
            R_hat,
            episode_return,
            lap_number,
            lap_time,
            completed_episodes,
        ) = batch

        training_config = self.config.training
        optimizer = cast(torch.optim.Optimizer, self.optimizers())

        log_p_new, H, V_phi = self.policy.evaluate_actions(s, a)
        loss_dict = ppo_loss(
            log_p_new=log_p_new,
            log_p_cur=log_p_cur,
            A_hat=A_hat,
            R_hat=R_hat,
            V_phi=V_phi,
            H=H,
            epsilon=training_config.epsilon,
            c1=training_config.c1,
            c2=training_config.c2,
        )

        loss = loss_dict["loss"]
        optimizer.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.parameters(), training_config.max_grad_norm)
        optimizer.step()

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/L_PPO", loss_dict["L_PPO"], on_step=False, on_epoch=True)
        self.log("train/L_clip", loss_dict["L_clip"], on_step=False, on_epoch=True)
        self.log("train/L_V", loss_dict["L_V"], on_step=False, on_epoch=True)
        self.log("train/L_H", loss_dict["L_H"], on_step=False, on_epoch=True)
        self.log("train/rho", loss_dict["rho"], on_step=False, on_epoch=True)
        self.log(
            "train/approx_kl", loss_dict["approx_kl"], on_step=False, on_epoch=True
        )
        self.log(
            "train/clip_fraction",
            loss_dict["clip_fraction"],
            on_step=False,
            on_epoch=True,
        )
        self.log("train/episode_return", episode_return, on_step=False, on_epoch=True)
        self.log("train/lap_number", lap_number, on_step=False, on_epoch=True)
        self.log("train/lap_time", lap_time, on_step=False, on_epoch=True)
        self.log(
            "train/completed_episodes",
            completed_episodes,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:  # noqa: D102
        return torch.optim.Adam(
            self.parameters(), lr=self.config.training.learning_rate
        )


# ═══════════════════════════════════════════════════════════════════════════════
# compute_gae – Generalized Advantage Estimation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_gae(
    r: Tensor,
    V_phi: Tensor,
    terminal: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    return _compiled_compute_gae(r, V_phi, terminal, gamma, lam)


def _compute_gae_recurrence(
    r: Tensor,
    V_phi: Tensor,
    terminal: Tensor,
    gamma: float,
    lam: float,
) -> Tuple[Tensor, Tensor]:
    """Compute ``A_hat`` and ``R_hat`` from Algorithm 1.

    Fixed format: ``r`` and ``terminal`` are ``(M, n)`` tensors, and
    ``V_phi`` is ``(M, n + 1)`` with the bootstrap value in the final column.
    Common names: ``r`` = rewards, ``terminal`` = done flags,
    ``A_hat`` = advantages, ``R_hat`` = returns.

    Args:
        r:           ``r(s_{j,t}, a_{j,t})`` with shape ``(M, n)``.
        V_phi:       ``V_phi(s_{j,t})`` with shape ``(M, n + 1)``.
        terminal:    Terminal/done flags with shape ``(M, n)``.
        gamma:       Discount factor.
        lam:         GAE ``lambda`` parameter.

    Returns:
        ``(A_hat, R_hat)`` tensors.
    """
    M: int = r.shape[0]  # noqa: N806
    n: int = r.shape[1]

    V_phi_s_j_n: Tensor = V_phi[:, -1]
    V_phi_s_j_t: Tensor = V_phi[:, :-1]

    A_hat = torch.zeros((M, n), dtype=r.dtype, device=r.device)
    A_hat_next = torch.zeros(M, dtype=r.dtype, device=r.device)

    for t in range(n - 1, -1, -1):
        non_terminal = 1.0 - terminal[:, t]
        V_phi_s_next = V_phi_s_j_n if t == n - 1 else V_phi_s_j_t[:, t + 1]
        delta_V_phi = r[:, t] + gamma * V_phi_s_next * non_terminal - V_phi_s_j_t[:, t]
        A_hat[:, t] = delta_V_phi + gamma * lam * non_terminal * A_hat_next
        A_hat_next = A_hat[:, t]

    R_hat = A_hat + V_phi_s_j_t

    return A_hat, R_hat


_compiled_compute_gae = torch.compile(_compute_gae_recurrence)


# ═══════════════════════════════════════════════════════════════════════════════
# ppo_loss – Clipped-surrogate PPO objective
# ═══════════════════════════════════════════════════════════════════════════════


def ppo_loss(
    log_p_new: Tensor,
    log_p_cur: Tensor,
    A_hat: Tensor,
    R_hat: Tensor,
    V_phi: Tensor,
    H: Tensor,
    epsilon: float = 0.2,
    c1: float = 0.5,
    c2: float = 0.01,
) -> dict[str, Tensor]:
    """Compute the minibatch PPO objective from the documentation.

    This is the tensor version of Algorithm 2 in ``ref/ppo_documentation.tex``:

    ``rho = exp(log p_new(s, a) - log p_cur(s, a))``

    ``L_PPO = L_CLIP - c1 * L_V + c2 * L_H``

    PyTorch optimizers minimize, so the returned ``loss`` is ``-L_PPO``.

    Args:
        log_p_new:  ``log p_{theta_new}(s_{j,t}, a_{j,t})``.
        log_p_cur:  ``log p_{theta_cur}(s_{j,t}, a_{j,t})`` recorded during rollout.
        A_hat:      ``A_hat_{j,t}`` from GAE.
        R_hat:      ``R_hat_{j,t} = A_hat_{j,t} + V_phi(s_{j,t})``.
        V_phi:      Current value estimate ``V_phi(s_{j,t})``.
        H:          Entropy ``H(pi_{theta_new}(s_{j,t}, ·))``.
        epsilon:    Clipping parameter ``ε``.
        c1:         Value-loss coefficient.
        c2:         Entropy coefficient.

    Returns:
        Dictionary with keys:

        - ``loss``            ``-L_PPO`` scalar to minimise
        - ``L_clip``          clipped surrogate objective value
        - ``L_V``             value MSE objective
        - ``L_H``             mean entropy objective
        - ``rho``             mean policy ratio diagnostic
        - ``approx_kl``       clamped ``mean(log_p_cur − log_p_new)`` (detached)
        - ``clip_fraction``   fraction of samples where ``|rho − 1| > ε``
    """
    log_ratio = torch.clamp(log_p_new - log_p_cur, -20.0, 20.0)
    rho = torch.exp(log_ratio)

    unclipped = rho * A_hat
    clipped = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon) * A_hat
    L_clip = torch.min(unclipped, clipped).mean()

    L_V = F.mse_loss(V_phi, R_hat)
    L_H = H.mean()

    L_PPO = L_clip - c1 * L_V + c2 * L_H
    loss = -L_PPO

    with torch.no_grad():
        approx_kl = (-log_ratio).mean()
        clip_fraction = ((rho - 1.0).abs() > epsilon).float().mean()

    return {
        "loss": loss,
        "L_PPO": L_PPO.detach(),
        "L_clip": L_clip.detach(),
        "L_V": L_V.detach(),
        "L_H": L_H.detach(),
        "rho": rho.detach().mean(),
        # Backward-compatible metric aliases for Lightning logs/users.
        "policy_loss": (-L_clip).detach(),
        "value_loss": L_V.detach(),
        "entropy_loss": L_H.detach(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
