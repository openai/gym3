import gym
import numpy as np
import pytest
from baselines.common.vec_env import DummyVecEnv

import gym3
from gym3.types_np import multimap, sample, zeros


def gym3_rollout(e):
    for _ in range(10):
        rew, ob, done = e.observe()
        print(multimap(lambda x: x.shape, ob), rew.shape, done.shape)
        e.act(sample(e.ac_space, (e.num,)))


def gym_rollout(e):
    done = True
    for _ in range(10):
        if done:
            e.reset()
        ob, rew, done, _info = e.step(e.action_space.sample())
        print(ob.shape, rew.shape, done.shape)


def vecenv_rollout(e):
    e.reset()
    for _ in range(10):
        ob, rew, done, _info = e.step(
            np.stack([e.action_space.sample() for _ in range(e.num_envs)])
        )
        print(ob.shape, rew.shape, done.shape)


ENV_IDS = ["CartPole-v1", "Pendulum-v0"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_from_gym_env(env_id):
    gym_env = gym.make(env_id)
    e = gym3.FromGymEnv(gym_env)
    gym3_rollout(e)
    obs = e.callmethod("seed", [0])
    assert len(obs) == 1


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_from_baselines_env(env_id):
    env_fn = lambda: gym.make(env_id)
    e = gym3.FromBaselinesVecEnv(DummyVecEnv([env_fn]))
    gym3_rollout(e)


def test_from_procgen_env():
    e = gym3.testing.AssertSpacesWrapper(gym3.testing.FixedSequenceEnv(num=3))
    gym3_rollout(e)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_to_gym_env(env_id):
    e = gym3.ToGymEnv(gym3.FromGymEnv(gym.make(env_id)))
    gym_rollout(e)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_to_baselines_env(env_id):
    e = gym3.ToBaselinesVecEnv(gym3.FromGymEnv(gym.make(env_id)))
    vecenv_rollout(e)


def test_vectorize_gym():
    env = gym3.vectorize_gym(num=2, env_fn=gym.make, env_kwargs={"id": "Pendulum-v0"})
    env.observe()
    env.act(zeros(env.ac_space, bshape=(env.num,)))
    env.observe()
