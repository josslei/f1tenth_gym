"""Tensor batch containers for PPO updates."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import TensorDataset


@dataclass
class PPOBatch:
    """Canonical minibatch fields from the document's index set ``I``.

    Common names: ``s`` = observations/states, ``a`` = actions,
    ``log_p_cur`` = old log probabilities, ``A_hat`` = advantages,
    ``R_hat`` = returns.
    """

    s: Tensor
    a: Tensor
    log_p_cur: Tensor
    A_hat: Tensor
    R_hat: Tensor


def make_update_dataset(
    s: Tensor,
    a: Tensor,
    log_p_cur: Tensor,
    A_hat: Tensor,
    R_hat: Tensor,
) -> TensorDataset:
    """Wrap tensor fields as ``(s, a, log_p_cur, A_hat, R_hat)`` samples.

    Common names: ``s`` = observations/states, ``a`` = actions,
    ``log_p_cur`` = old log probabilities, ``A_hat`` = advantages,
    ``R_hat`` = returns.
    """
    return TensorDataset(s, a, log_p_cur, A_hat, R_hat)


__all__ = ["PPOBatch", "make_update_dataset"]
