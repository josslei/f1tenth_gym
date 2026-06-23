from typing import Any

import torch

class MuZeroSearch:
    def __init__(
        self,
        model: Any,
        num_iters: int,
        temperature: float,
        c_puct: float,
        batch_size: int,
        action_count: int,
        hidden_size: int,
        max_nodes: int = 0,
        device: torch.device | None = None,
        print_metrics: bool = False,
    ) -> None: ...
    def search_batch(self, obs_batch: torch.Tensor) -> torch.Tensor: ...
    def search_one(self, obs: torch.Tensor) -> torch.Tensor: ...
    def get_metrics(self) -> dict[str, float]: ...
