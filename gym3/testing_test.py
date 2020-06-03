import time

from gym3 import types, types_np
from gym3.testing import IdentityEnv, TimingEnv


def test_fast_env():
    num_env = 2
    num_steps = 10000
    episode_len = 100
    start = time.time()
    env = TimingEnv(num=num_env, episode_len=episode_len)
    episode_count = 0
    expected_episode_count = num_env * num_steps / episode_len
    for i in range(num_steps):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        _rew, _obs, first = env.observe()
        for f in first:
            if f:
                episode_count += 1

        if i == num_steps - 2:
            assert episode_count == expected_episode_count - num_env
    elapsed = time.time() - start
    assert elapsed / num_steps < 1e-3
    assert episode_count == expected_episode_count


def test_identity_env():
    env = IdentityEnv(space=types.discrete_scalar(1024), episode_len=2)
    rew, ob, first = env.observe()
    assert ob == ob
    assert first
    assert rew == 0
    env.act(ob)
    rew, ob, first = env.observe()
    assert not first
    assert rew == 1
    env.act(ob)
    rew, ob, first = env.observe()
    assert first
    assert rew == 1


def test_identity_env_delay():
    delay = 2
    for space in [types.discrete_scalar(1024)]:
        env = IdentityEnv(space=space, episode_len=delay * 2, delay_steps=delay)
        _rew, obs, _first = env.observe()
        obs_queue = [obs]
        for i in range(delay):
            env.act(0)
            _rew, obs, _first = env.observe()
            obs_queue.append(obs)

        first = False
        for i in range(delay):
            env.act(obs_queue.pop(0))
            rew, obs, first = env.observe()
            obs_queue.append(obs)
            assert rew == 1
        assert first
