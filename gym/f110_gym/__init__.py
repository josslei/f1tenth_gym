from f110_gym.envs import F110Env
from gymnasium.envs.registration import register

register(
    id="f110-v0",
    entry_point="f110_gym:F110Env",
)
