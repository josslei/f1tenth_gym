from pathlib import Path

import torch.nn as nn

from models.policies import GaussianMLPPolicy, Policy, make_policy
from models.ppo import PPOConfig, PolicyConfig


def test_default_ppo_config_selects_gaussian_mlp_policy():
    config = PPOConfig.from_yaml(Path("configs/ppo/default.yaml"))

    assert config.policy.name == "GaussianMLPPolicy"
    assert config.policy.kwargs == {"hidden_size": 256}


def test_make_policy_instantiates_configured_policy():
    policy_config = PolicyConfig(
        name="GaussianMLPPolicy",
        kwargs={"hidden_size": 64},
    )

    policy = make_policy(policy_config, obs_dim=10, action_dim=2)

    assert isinstance(policy, GaussianMLPPolicy)
    assert isinstance(policy, Policy)
    assert policy.obs_dim == 10
    assert policy.action_dim == 2
    assert isinstance(policy.net[0], nn.Linear)
    assert policy.net[0].out_features == 64


def test_with_policy_kwargs_merges_without_mutating_original():
    config = PPOConfig.from_yaml(Path("configs/ppo/default.yaml"))

    updated = config.with_policy_kwargs(hidden_size=32)

    assert config.policy.kwargs == {"hidden_size": 256}
    assert updated.policy.kwargs == {"hidden_size": 32}
