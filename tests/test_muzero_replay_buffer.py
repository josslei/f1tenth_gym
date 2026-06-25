import numpy as np

from models.muzero.replay_buffer import MuZeroReplayBuffer, MuZeroTransition


def test_n_step_return_bootstraps_from_root_value():
    buffer = MuZeroReplayBuffer(max_size=8, unroll_steps=2, td_steps=2, discount=0.5)
    policy = np.array([1.0, 0.0], dtype=np.float32)
    trajectory = [
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=1.0,
            done=False,
            root_policy=policy,
            root_value=10.0,
        ),
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=2.0,
            done=False,
            root_policy=policy,
            root_value=20.0,
        ),
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=3.0,
            done=False,
            root_policy=policy,
            root_value=30.0,
        ),
    ]

    assert buffer._n_step_return(trajectory, 0) == 1.0 + 0.5 * 2.0 + 0.25 * 30.0


def test_n_step_return_does_not_bootstrap_past_terminal():
    buffer = MuZeroReplayBuffer(max_size=8, unroll_steps=2, td_steps=2, discount=0.5)
    policy = np.array([1.0, 0.0], dtype=np.float32)
    trajectory = [
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=1.0,
            done=False,
            root_policy=policy,
            root_value=10.0,
        ),
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=2.0,
            done=True,
            root_policy=policy,
            root_value=20.0,
        ),
        MuZeroTransition(
            obs=np.zeros(1, dtype=np.float32),
            action=0,
            reward=3.0,
            done=False,
            root_policy=policy,
            root_value=30.0,
        ),
    ]

    assert buffer._n_step_return(trajectory, 0) == 1.0 + 0.5 * 2.0
