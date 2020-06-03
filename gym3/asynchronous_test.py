import time

import pytest

from gym3 import AsynchronousWrapper, Wrapper
from gym3.testing import IdentityEnv, TimingEnv


class CustomException(Exception):
    pass


class ExceptionOnAct(Wrapper):
    def act(self, ac):
        raise CustomException("oh no")


class RecordActs(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.acts = []

    def act(self, ac):
        self.env.act(ac)
        self.acts.append(ac)


def test_asynchronous():
    delay = 0.1
    num_steps = 10
    env = TimingEnv(delay_seconds=delay)

    start = time.time()
    for i in range(num_steps):
        env.act(0)
    assert time.time() - start > delay * num_steps

    async_env = AsynchronousWrapper(env)
    start = time.time()
    for i in range(num_steps):
        async_env.act(0)
    assert time.time() - start < 0.1
    async_env.observe()
    assert time.time() - start > delay * num_steps


def test_asynchronous_ordering():
    num_steps = 100
    env = IdentityEnv()
    env = RecordActs(env)
    env = AsynchronousWrapper(env)
    for i in range(num_steps):
        env.act(i)
    env.observe()
    assert env.env.acts == list(range(num_steps))


def test_asynchronous_exception():
    env = IdentityEnv()
    env = ExceptionOnAct(env)
    env = AsynchronousWrapper(env)
    env.observe()
    env.act(0)
    with pytest.raises(CustomException):
        env.observe()
