"""
Adapters to convert to/from gym and baselines VecEnv interfaces

Late imports and string annotations are used so that if gym is not installed,
there are no errors when importing gym3.
"""
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from gym3 import types_np
from gym3.concat import ConcatEnv
from gym3.env import Env
from gym3.internal import misc
from gym3.subproc import SubprocEnv
from gym3.types import (
    DictType,
    Discrete,
    Real,
    TensorType,
    ValType,
    multimap,
    INTEGER_DTYPE_NAMES,
    FLOAT_DTYPE_NAMES,
)


def _space2vt(space: "gym.spaces.Space"):
    from gym import spaces

    if isinstance(space, spaces.Box):
        if space.dtype.name in INTEGER_DTYPE_NAMES:
            high = space.high.flat[0]
            assert (space.low == 0).all() and (
                space.high == high
            ).all(), "only identical high/low bounds across all dimensions are supported, and low must be 0 for integer types, please wrap your environment to adjust your Box space bounds, split into a Dict space, or use float32 as the dtype"
            return TensorType(
                shape=space.shape,
                eltype=Discrete(n=high + 1, dtype_name=space.dtype.name),
            )
        else:
            assert (
                space.dtype.name in FLOAT_DTYPE_NAMES
            ), f"only {FLOAT_DTYPE_NAMES} is supported for real values, please wrap your environment so that a valid dtype is used"
            return TensorType(shape=space.shape, eltype=Real(dtype_name=space.dtype.name))
    elif isinstance(space, spaces.Discrete):
        return TensorType(
            shape=(), eltype=Discrete(space.n, dtype_name=space.dtype.name)
        )
    elif isinstance(space, spaces.MultiDiscrete):
        assert misc.allsame(
            space.nvec
        ), f"only multidiscrete with identical values of n is allowed, please wrap your environment so that it has a Dict space with individual Discrete spaces for each dimension instead"
        return TensorType(
            shape=(len(space.nvec),),
            eltype=Discrete(space.nvec[0], dtype_name=space.dtype.name),
        )
    elif isinstance(space, spaces.MultiBinary):
        return TensorType(
            shape=(space.n,), eltype=Discrete(2, dtype_name=space.dtype.name)
        )
    elif isinstance(space, spaces.Dict):
        return DictType(
            **{name: _space2vt(subspace) for (name, subspace) in space.spaces.items()}
        )
    elif isinstance(space, spaces.Tuple):
        assert (
            False
        ), "tuple space not supported, please wrap your environment so that it has a Dict space instead of a Tuple space"
    else:
        raise NotImplementedError


def _vt2space(vt: ValType):
    from gym import spaces

    def tt2space(tt: TensorType):
        if isinstance(tt.eltype, Discrete):
            if tt.ndim == 0:
                return spaces.Discrete(tt.eltype.n)
            else:
                return spaces.Box(
                    low=0,
                    high=tt.eltype.n - 1,
                    shape=tt.shape,
                    dtype=types_np.dtype(tt),
                )
        elif isinstance(tt.eltype, Real):
            return spaces.Box(
                shape=tt.shape,
                dtype=types_np.dtype(tt),
                low=float("-inf"),
                high=float("inf"),
            )
        else:
            raise NotImplementedError

    space = multimap(tt2space, vt)

    def dict2dict_space(d):
        if isinstance(d, dict):
            return spaces.Dict({k: dict2dict_space(v) for k, v in d.items()})
        else:
            return d

    return dict2dict_space(space)


def _assert_num_envs_1(ac):
    """ Check that the action is consistent with num_envs = 1 """
    if isinstance(ac, dict):
        for a in ac.values():
            _assert_num_envs_1(a)
    else:
        assert len(ac) == 1


class FromGymEnv(Env):
    """
    Create a gym3 environment from a gym environment.
    
    Notes:
        * low/high values for continuous spaces will be discarded since gym3 does not support these
        * some spaces are not supported
        * `callmethod()` will call methods on the underlying gym environment

    :param gym_env: gym environment to adapt
    :param render_mode: if set, gym_env.render() will be called each time env.act() is called with the value
            used as the `mode` argument.
            If render_mode == "rgb_array", the return value will be placed in the info dict, which
            you can retrieve with env.get_info()["rgb"].
    """

    def __init__(self, gym_env: Env, render_mode=None, seed=None):
        super().__init__(
            ob_space=_space2vt(gym_env.observation_space),
            ac_space=_space2vt(gym_env.action_space),
            num=1,
        )
        gym_env.seed(seed)
        self.gym_env = gym_env
        self.last_ob = gym_env.reset()
        self.last_rew = 0.0
        self.last_first = True
        self.render_mode = render_mode
        self.info = {}

    def observe(self) -> Tuple[Any, Any, Any]:
        return (
            np.array([self.last_rew], "f"),
            multimap(lambda val: np.expand_dims(np.array(val), axis=0), self.last_ob),
            np.array([self.last_first], bool),
        )

    def get_info(self) -> List[Dict]:
        self.observe()
        return [self.info]

    def act(self, ac: Any) -> None:
        # Check we got an action consistent with num_envs=1
        _assert_num_envs_1(ac)
        aczero = multimap(lambda x: x[0], ac)
        self.last_ob, self.last_rew, self.last_first, self.info = self.gym_env.step(
            aczero
        )
        if self.render_mode == "rgb_array":
            self.info["rgb"] = self.gym_env.render(mode="rgb_array")
        elif self.render_mode is not None:
            self.gym_env.render(mode=self.render_mode)
        if self.last_first:
            self.last_ob = self.gym_env.reset()

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        call_args = [arg[0] for arg in args]
        call_kwargs = {k: v[0] for k, v in kwargs.items()}
        return [getattr(self.gym_env, method)(*call_args, **call_kwargs)]


class FromBaselinesVecEnv(Env):
    """
    Create a gym3 environment from a baselines VecEnv environment.  baselines VecEnv is rarely used outside
    of the baselines code, and is not recommended for new environments.
    
    Notes:
        * low/high values for continuous spaces will be discarded since gym3 does not support these
        * `callmethod()` will call methods on the underlying VecEnv environment

    :param bv_env: baselines VecEnv environment to adapt
    :param render_mode: if set, bv_env.render() will be called each time env.observe() is called for the first
            time after an action with the value used as the `mode` argument.
            If render_mode == "rgb_array", the return value will be placed in the info dict, which
            you can retrieve with env.get_info()["rgb"].
    """

    def __init__(self, bv_env: "baselines.common.VecEnv", render_mode=None):
        super().__init__(
            ob_space=_space2vt(bv_env.observation_space),
            ac_space=_space2vt(bv_env.action_space),
            num=bv_env.num_envs,
        )
        self.observe_tuple = None
        self.bv_env = bv_env
        self.observe_tuple = (
            np.zeros(self.num, "f"),
            self.bv_env.reset(),
            np.zeros(self.num, np.bool),
        )
        self.have_new_action = False
        self.info = [{} for _ in range(self.num)]
        self.render_mode = render_mode
        # If have_new_action, next call must be step_wait
        # else, next call must be step_async

    def observe(self) -> Tuple[Any, Any, Any]:
        if self.have_new_action:
            ob, rew, first, self.info = self.bv_env.step_wait()
            self.have_new_action = False
            self.observe_tuple = (rew, ob, first)
            if self.render_mode == "rgb_array":
                renders = self.bv_env.get_images()
                for idx, img in enumerate(renders):
                    self.info[idx]["rgb"] = img
            elif self.render_mode is not None:
                self.bv_env.render(mode=self.render_mode)
        return self.observe_tuple

    def get_info(self) -> List[Dict]:
        self.observe()
        return self.info

    def act(self, ac: Any) -> None:
        if self.have_new_action:
            self.bv_env.step_wait()
        self.bv_env.step_async(ac)
        self.have_new_action = True

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return getattr(self.bv_env, method)(*args, **kwargs)


class ToGymEnv:
    """
    Create a gym environment from a gym3 environment.

    Notes:
        * The `render()` method does nothing in "human" mode, in "rgb_array" mode the info dict is checked
            for a key named "rgb" and info["rgb"][0] is returned if present
        * `seed()` and `close() are ignored since gym3 environments do not require these methods
        * `reset()` is ignored if used before an episode is complete because gym3 environments
            reset automatically, if `reset()` was called before the end of an episode, a warning is printed
    
    :param env: gym3 environment to adapt
    """

    def __init__(self, env):
        self.env = env
        assert env.num == 1
        self.observation_space = _vt2space(env.ob_space)
        self.action_space = _vt2space(env.ac_space)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first[0]:
            print("Warning: early reset ignored")
        return multimap(lambda x: x[0], ob)

    def step(self, ac):
        _, prev_ob, _ = self.env.observe()
        self.env.act(np.array([ac]))
        rew, ob, first = self.env.observe()
        if first[0]:
            ob = prev_ob
        return multimap(lambda x: x[0], ob), rew[0], first[0], self.env.get_info()[0]

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def seed(self):
        print("Warning: seed ignored")

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self


class ToBaselinesVecEnv:
    """
    Create a baselines VecEnv environment from a gym3 environment.

    :param env: gym3 environment to adapt
    """

    def __init__(self, env):
        self.env = env
        self.observation_space = _vt2space(env.ob_space)
        self.action_space = _vt2space(env.ac_space)

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first.all():
            print("Warning: manual reset ignored")
        return ob

    def step_async(self, ac):
        self.env.act(ac)

    def step_wait(self):
        rew, ob, first = self.env.observe()
        return ob, rew, first, self.env.get_info()

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self.env.num

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def close(self):
        pass


def _make_gym_env(env_fn, env_kwargs, render_mode=None, seed=None):
    gym_env = env_fn(**env_kwargs)
    env = FromGymEnv(gym_env, render_mode=render_mode, seed=seed)
    return env


def vectorize_gym(
    num, env_fn=None, env_kwargs=None, use_subproc=True, render_mode=None, seed=None
):
    """
    Given a function that creates a gym environment and a number of environments to create,
    create the environments in subprocesses and combine them into a single gym3 Env.

    This is meant as a replacement for baselines' SubprocVecEnv and DummyVecEnv

    If you want to use this for a registered gym env, the default is to use gym.make as the function
    to call:

        env = vectorize_gym(num=2, env_kwargs={"id": "Pendulum-v0"})
    
    :param num: number of gym environments to create
    :param env_fn: function to call to create the gym environment, defaults to `gym.make` 
    :param env_kwargs: keyword arguments to pass to env_fn
    :param use_subproc: if set to False, create the environment in the current process
    :param render_mode: if set, this will be passed to the `FromGymEnv` adapter,
        see the documentation for `FromGymEnv` for more information
    """
    if env_fn is None:
        import gym

        env_fn = gym.make
    if env_kwargs is None:
        env_kwargs = {}
    if use_subproc:
        envs = [
            SubprocEnv(
                env_fn=_make_gym_env,
                env_kwargs=dict(
                    env_fn=env_fn, env_kwargs=env_kwargs, render_mode=render_mode, seed=seed,
                ),
            )
            for _ in range(num)
        ]
    else:
        envs = [
            _make_gym_env(env_fn=env_fn, env_kwargs=env_kwargs, render_mode=render_mode, seed=seed)
            for _ in range(num)
        ]
    return ConcatEnv(envs)
