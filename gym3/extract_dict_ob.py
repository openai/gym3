from gym3.wrapper import Wrapper


class ExtractDictObWrapper(Wrapper):
    def __init__(self, env, key):
        self._key = key
        super().__init__(env, ob_space=env.ob_space[self._key])

    def observe(self):
        rew, ob, done = super().observe()
        return rew, ob[self._key], done
