import numpy as np
import pytest

from gym3 import types
from gym3.concat import ConcatEnv
from gym3.testing import AssertSpacesWrapper, IdentityEnv
from gym3.wrapper import Wrapper


class AddInfo(Wrapper):
    def __init__(self, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = id

    def get_info(self):
        infos = [info.copy() for info in self.env.get_info()]
        for info in infos:
            info["id"] = self._id
        return infos


@pytest.mark.parametrize("space_type", ["binary", "dict", "real_dict"])
def test_concat(space_type):
    if space_type == "binary":
        space = types.discrete_scalar(2)
    elif space_type == "dict":
        space = types.DictType(degrees=types.discrete_scalar(360))
    elif space_type == "real_dict":
        space = types.DictType(degrees=types.TensorType(shape=(), eltype=types.Real()))
    else:
        raise Exception(f"invalid space_type {space_type}")

    base_env1 = IdentityEnv(space=space, episode_len=1, seed=0)
    base_env1.f = lambda x: [x[0] ** 2]
    base_env2 = IdentityEnv(space=space, episode_len=2, seed=1)
    base_env2.f = lambda x: [2 * x[0]]
    env1 = AssertSpacesWrapper(base_env1)
    env1 = AddInfo(env=env1, id=1)
    env2 = AssertSpacesWrapper(base_env2)
    env2 = AddInfo(env=env2, id=2)
    env = AssertSpacesWrapper(ConcatEnv([env1, env2]))
    rew, ob, first = env.observe()
    assert np.array_equal(rew, np.array([0, 0]))
    if isinstance(space, types.DictType):
        ob = ob["degrees"]
    assert ob.shape == (2,)
    assert ob[0] != ob[1]
    assert np.array_equal(first, np.array([1, 1]))
    act = np.array([ob[0], ob[0]])
    if isinstance(space, types.DictType):
        act = dict(degrees=act)
    env.act(act)
    rew, _ob, first = env.observe()
    if space_type == "real_dict":
        assert rew[0] == 0
        assert rew[1] < 0
    else:
        assert np.array_equal(rew, np.array([1, 0]))
    assert np.array_equal(first, np.array([1, 0]))
    assert env.get_info() == [{"id": 1}, {"id": 2}]
    with pytest.raises(AssertionError):
        env.callmethod("f", [2, 3, 4])

    assert env.callmethod("f", [2, 3]) == [2 ** 2, 2 * 3]
