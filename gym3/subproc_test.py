import sys
import platform

import numpy as np
import pytest
import retro

from gym3.asynchronous_test import CustomException, ExceptionOnAct
from gym3.concat import ConcatEnv
from gym3.internal.test_with_mpi import run_test_with_mpi
from gym3.interop import FromGymEnv
from gym3.subproc import SubprocEnv, SubprocError
from gym3.testing import IdentityEnv
from gym3.wrapper import Wrapper


class StepInfo(Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self._step = 0

    def act(self, ac):
        self.env.act(ac)
        self._step += 1

    def get_info(self):
        infos = [info.copy() for info in self.env.get_info()]
        for info in infos:
            info["step"] = self._step
        return infos


def create_test_env(episode_len):
    env = IdentityEnv(episode_len=episode_len)
    env = StepInfo(env)
    return env


def test_subproc():
    env = SubprocEnv(env_fn=create_test_env, env_kwargs=dict(episode_len=2))
    rew, obs, first = env.observe()
    assert first
    assert rew == 0.0
    env.act(obs)
    env.act(obs)
    env.act(obs)
    infos = env.get_info()
    assert infos[0]["step"] == 3


def create_callmethod_test_env():
    env = IdentityEnv()
    env.square = lambda x: x ** 2
    return env


def test_subproc_callmethod():
    env = SubprocEnv(env_fn=create_callmethod_test_env)
    assert env.callmethod("square", 2) == 4


def create_test_env_exception_on_create():
    raise CustomException("oh no")


def create_test_env_exception_on_act():
    env = IdentityEnv()
    env = ExceptionOnAct(env)
    return env


def create_test_env_exit_on_create():
    sys.exit(1)


class ExitOnAct(Wrapper):
    def act(self, ac):
        sys.exit(1)


def create_test_env_exit_on_act():
    env = IdentityEnv()
    env = ExitOnAct(env)
    return env


def test_subproc_exception():
    with pytest.raises(SubprocError):
        env = SubprocEnv(env_fn=create_test_env_exception_on_create)

    env = SubprocEnv(env_fn=create_test_env_exception_on_act)
    rew, obs, first = env.observe()
    env.act(obs)
    with pytest.raises(SubprocError):
        # because act does not get the return value, the error actually appears during the
        # next call to observe()
        env.observe()

    with pytest.raises(SubprocError):
        env = SubprocEnv(env_fn=create_test_env_exit_on_create)

    env = SubprocEnv(env_fn=create_test_env_exit_on_act)
    rew, obs, first = env.observe()
    env.act(obs)
    with pytest.raises(SubprocError):
        env.observe()


def create_test_env_mpi(episode_len):
    # this is necessary for the MPI test to fail
    from mpi4py import MPI

    env = IdentityEnv(episode_len=episode_len)
    return env


def subproc_mpi():
    # this is necessary for the MPI test to fail
    from mpi4py import MPI

    env = SubprocEnv(env_fn=create_test_env_mpi, env_kwargs=dict(episode_len=2))
    rew, obs, first = env.observe()


@pytest.mark.skipif(platform.system() == "Windows", reason="microsoft mpi + mpi4py not installed")
# https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
def test_subproc_mpiexec():
    run_test_with_mpi("gym3.subproc_test:subproc_mpi")


def test_subproc_cloudpickle():
    def make_env():
        env = IdentityEnv()
        env.square = lambda x: x ** 2
        return env

    env = SubprocEnv(env_fn=make_env)
    assert env.callmethod("square", 2) == 4


def make_retro_env(**kwargs):
    gym_env = retro.make(**kwargs)
    env = FromGymEnv(gym_env)
    return env


def test_subproc_retro():
    env = retro.make("Airstriker-Genesis")
    with pytest.raises(RuntimeError):
        env = retro.make("Airstriker-Genesis")

    envs = [
        SubprocEnv(env_fn=make_retro_env, env_kwargs=dict(game="Airstriker-Genesis"))
        for _ in range(2)
    ]
    env = ConcatEnv(envs)
    rew, ob, first = env.observe()
    assert np.array_equal(ob[0], ob[1])
