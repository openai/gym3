from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from gym3.env import Env
from gym3.internal import misc
from gym3.types_np import concat, split


def _chunk_seq(x: Sequence, sizes: Sequence[int]) -> Sequence[Any]:
    result = []
    assert len(x) == sum(sizes), f"x has incorrect length {len(x)} != {sum(sizes)}"
    start = 0
    for size in sizes:
        end = start + size
        result.append(x[start:end])
        start = end
    return result


class ConcatEnv(Env):
    """
    Concatenate multiple environments into a single environment.

    :param envs: list of environments to concatenate, must all have the same ac_space and ob_space
    """

    def __init__(self, envs: Sequence[Env]):
        total_num = sum(env.num for env in envs)
        super().__init__(envs[0].ob_space, envs[0].ac_space, total_num)
        assert misc.allsame([env.ac_space for env in envs])
        assert misc.allsame([env.ob_space for env in envs])
        self.envs = envs

    def observe(self) -> Tuple[Any, Any, Any]:
        rews, obs, firsts = zip(*[env.observe() for env in self.envs])
        return np.concatenate(rews), concat(obs), np.concatenate(firsts)

    def get_info(self) -> List[Dict]:
        result = []
        for env in self.envs:
            result.extend(env.get_info())
        return result

    def act(self, ac: Any) -> None:
        split_ac = split(ac, sections=np.cumsum([env.num for env in self.envs]))
        for env, a in zip(self.envs, split_ac):
            env.act(a)

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        sizes = [env.num for env in self.envs]
        chunked_args = [_chunk_seq(arg, sizes) for arg in args]
        chunked_kwargs = {k: _chunk_seq(v, sizes) for k, v in kwargs.items()}
        result = []
        for chunk_idx, env in enumerate(self.envs):
            env_args = [arg[chunk_idx] for arg in chunked_args]
            env_kwargs = {k: v[chunk_idx] for k, v in chunked_kwargs.items()}
            result.extend(env.callmethod(method, *env_args, **env_kwargs))
        return result
