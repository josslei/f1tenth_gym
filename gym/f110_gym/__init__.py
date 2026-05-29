from gymnasium.envs.registration import register
from f110_gym.envs.f110_env import F110Env

__all__ = ["F110Env"]

register(
    id="f110-v0",
    entry_point="f110_gym:F110Env",
)
