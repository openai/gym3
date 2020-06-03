from typing import Any, Dict, List, Sequence, Tuple

from gym3.env import Env


class Wrapper(Env):
    """
    An interface for reinforcement learning environments.

    :param env: gym3 environment to wrap
    :param ob_space: observation space to use (overrides env's observation space)
    :param ac_space: action space to use (overrides env's action space)
    """

    def __init__(self, env, ob_space=None, ac_space=None):
        super().__init__(ob_space or env.ob_space, ac_space or env.ac_space, env.num)
        self.env = env

    def observe(self) -> Tuple[Any, Any, Any]:
        return self.env.observe()

    def get_info(self) -> List[Dict]:
        return self.env.get_info()

    def act(self, ac: Any) -> None:
        return self.env.act(ac)

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return self.env.callmethod(method, *args, **kwargs)


def unwrap(env):
    """
    :param env: a gym3 environment that may have wrappers applied to it

    :returns: the gym3 environment without the wrappers
    """
    while isinstance(env, Wrapper):
        env = env.env
    return env
