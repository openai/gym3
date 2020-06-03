import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from gym3 import types, types_np
from gym3.env import Env
from gym3.wrapper import Wrapper


class AssertSpacesWrapper(Wrapper):
    """
    Assert that observations and actions match the spaces of the environment
    """

    def _assert_val_matches_space(
        self, space: types.TensorType, val: np.ndarray
    ) -> None:
        assert isinstance(val, np.ndarray), "action leaf is not a numpy array"
        assert val.shape[0] == self.num, "action batch dimension is incorrect"
        assert val.dtype == types_np.dtype(
            space
        ), f"value dtype {val.dtype} does not match space dtype: {types_np.dtype(space)}"
        if isinstance(space.eltype, types.Discrete):
            assert (0 <= val).all() and (val < space.eltype.n).all()

    def observe(self) -> Tuple[np.ndarray, Any, np.ndarray]:
        rew, ob, first = self.env.observe()
        types.multimap(self._assert_val_matches_space, self.ob_space, ob)
        assert rew.dtype is np.dtype(np.float32) and rew.shape == (self.num,)
        assert first.dtype is np.dtype(np.bool) and first.shape == (self.num,)
        return rew, ob, first

    def act(self, ac: Any) -> None:
        types.multimap(self._assert_val_matches_space, self.ac_space, ac)
        self.env.act(ac=ac)

    def get_info(self) -> List[Dict]:
        info = self.env.get_info()
        assert len(info) == self.num
        assert all(isinstance(x, dict) for x in info)
        return info


class IdentityEnv(Env):
    """
    An environment for testing where the observation at each step is the correct action
    to take on that step.

    :param space: observation/action space for the environment
    :param episode_len: steps per episode
    :param delay_steps: delay the correct action by this many steps
    :param seed: random seed used to determine observations
    """

    DEFAULT_TYPE = types.discrete_scalar(2)

    def __init__(
        self,
        space: types.ValType = DEFAULT_TYPE,
        episode_len: int = 1,
        delay_steps: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(ob_space=space, ac_space=space, num=1)
        self._seed = seed
        self._episode_len = episode_len
        self._step = None
        self._rews = np.zeros((1,), dtype=np.float32)
        self._firsts = np.zeros((1,), dtype=np.bool)
        self._delay_steps = delay_steps
        self._q = deque(maxlen=delay_steps + 1)
        self._rng = np.random.RandomState(seed)
        self._reset()

    def _reset(self) -> None:
        self._q.clear()
        for _ in range(self._delay_steps + 1):
            self._q.append(
                types_np.sample(self.ac_space, bshape=(self.num,), rng=self._rng)
            )
        self._step = 0
        self._firsts[0] = True

    def observe(self) -> Tuple[np.ndarray, Any, np.ndarray]:
        return self._rews, self._q[-1], self._firsts

    def act(self, ac: Any) -> None:
        self._firsts[0] = False
        state = self._q.popleft()

        rews = []

        def add_reward(subspace, substate, subval):
            if isinstance(subspace.eltype, types.Discrete):
                r = 1 if (substate == subval).all() else 0
            elif isinstance(subspace.eltype, types.Real):
                diff = subval - substate
                diff = diff[:]
                r = -0.5 * np.dot(diff, diff)
            else:
                raise Exception(f"unrecognized action space eltype {subspace.eltype}")
            rews.append(r)

        types.multimap(add_reward, self.ac_space, state, ac)
        rew = sum(rews) / len(rews)

        if self._step < self._delay_steps:
            # don't give any reward for guessing un-observed states
            rew = 0
        self._rews[0] = rew
        self._q.append(
            types_np.sample(self.ac_space, bshape=(self.num,), rng=self._rng)
        )
        self._step += 1
        if self._step >= self._episode_len:
            self._reset()


class TimingEnv(Env):
    """
    A fake environment for timing tests
    
    Useful for speed tests to show an upper bound on how fast your
    training can be if the environment is very fast.

    :param ob_space: observation space to use
    :param ac_space: action space to use
    :param num: number of parallel environments
    :param episode_len: steps per episode
    :param delay_seconds: sleep for this long when performing an action
    """

    def __init__(
        self,
        ob_space: types.ValType = types.TensorType(
            eltype=types.Discrete(256, dtype_name="uint8"), shape=(64, 64, 3)
        ),
        ac_space: types.ValType = types.discrete_scalar(2),
        num: int = 1,
        episode_len: int = 1000,
        delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(ob_space=ob_space, ac_space=ac_space, num=num)
        self._delay_seconds = delay_seconds
        self._episode_len = episode_len
        self._ob = types_np.zeros(self.ob_space, bshape=(self.num,))
        self._rews = np.zeros((self.num,), dtype=np.float32)
        self._steps = 0
        self._none_first = np.zeros((self.num,), dtype=np.bool)
        self._all_first = np.ones((self.num,), dtype=np.bool)
        self._infos = [{} for _ in range(self.num)]

    def observe(self) -> Tuple[np.ndarray, Any, np.ndarray]:
        if self._steps == 0:
            dones = self._all_first
        else:
            dones = self._none_first
        return self._rews, self._ob, dones

    def get_info(self) -> List[Dict]:
        return self._infos

    def act(self, ac: Any) -> None:  # pylint: disable=unused-argument
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)

        self._steps += 1
        if self._steps >= self._episode_len:
            self._steps = 0


class FixedSequenceEnv(Env):
    """
    The agent must guess a fixed sequence (same across all episodes and parallel environments)
    by taking a series of actions, with no observations to rely on.

    :param n_actions: number of actions available at each step
    :param episode_len: steps per episode
    :param num: number of parallel environments
    """

    def __init__(
        self, n_actions: int = 10, episode_len: int = 100, num: int = 1
    ) -> None:
        super().__init__(
            ac_space=types.discrete_scalar(n_actions),
            ob_space=types.discrete_scalar(1),
            num=num,
        )
        rng = np.random.RandomState(0)
        self.sequence = rng.randint(0, n_actions, size=episode_len)
        self.time = 0
        self.actions = np.zeros(num, "i")
        self.episode_len = episode_len

    def observe(self) -> Tuple[np.ndarray, Any, np.ndarray]:
        lastrew = (self.actions == self.sequence[self.time]).astype("f")
        ob = np.zeros(self.num, dtype=np.int64)
        first = np.full(fill_value=self.time % self.episode_len == 0, shape=(self.num,))
        return lastrew, ob, first

    def act(self, ac: Any) -> None:
        self.actions = ac
        self.time = (self.time + 1) % self.episode_len
