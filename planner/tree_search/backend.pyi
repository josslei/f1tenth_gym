import numpy as np

from numpy.typing import NDArray

class MuZeroSearch:
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
    def search_batch(self, obs_batch: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def search_one(self, obs: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def get_metrics(self) -> dict[str, float]: ...
