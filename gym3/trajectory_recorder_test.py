import os
import pickle
import tempfile
from glob import glob

import numpy as np

from gym3 import types, types_np
from gym3.concat import ConcatEnv
from gym3.testing import IdentityEnv
from gym3.trajectory_recorder import TrajectoryRecorderWrapper


def test_recorder():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep_len1 = 3
        ep_len2 = 4
        env1 = IdentityEnv(
            space=types.TensorType(eltype=types.Discrete(256), shape=(3, 3, 3)),
            episode_len=ep_len1,
        )
        env2 = IdentityEnv(
            space=types.TensorType(eltype=types.Discrete(256), shape=(3, 3, 3)),
            episode_len=ep_len2,
            seed=1,
        )
        env = ConcatEnv([env1, env2])
        env = TrajectoryRecorderWrapper(env=env, directory=tmpdir)
        _, obs, _ = env.observe()
        action = types_np.zeros(env.ac_space, bshape=(env.num,))
        action[1] = 1
        num_acs = 10
        for _ in range(num_acs):
            env.act(action)
        files = sorted(glob(os.path.join(tmpdir, "*.pickle")))
        print(files)
        assert len(files) == (num_acs // ep_len1) + (num_acs // ep_len2)

        with open(files[0], "rb") as f:
            loaded_traj = pickle.load(f)
        assert len(loaded_traj["ob"]) == ep_len1
        assert np.allclose(loaded_traj["ob"][0], obs[0])
        assert np.allclose(loaded_traj["act"][0], action[0])

        with open(files[1], "rb") as f:
            loaded_traj = pickle.load(f)
        assert len(loaded_traj["ob"]) == ep_len2
        assert np.allclose(loaded_traj["ob"][0], obs[1])
        assert np.allclose(loaded_traj["act"][0], action[1])


def test_recorder_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep_len1 = 3
        ep_len2 = 4
        space = types.DictType(
            a=types.TensorType(eltype=types.Discrete(256), shape=(2,)),
            b=types.DictType(
                c=types.TensorType(eltype=types.Discrete(256), shape=()),
                d=types.TensorType(eltype=types.Discrete(256), shape=(2, 2)),
            ),
        )
        env1 = IdentityEnv(space=space, episode_len=ep_len1)
        env2 = IdentityEnv(space=space, episode_len=ep_len2, seed=1)
        env = ConcatEnv([env1, env2])
        env = TrajectoryRecorderWrapper(env=env, directory=tmpdir)
        print(env._trajectories)
        _, obs, _ = env.observe()
        action = {
            "a": types_np.zeros(env.ac_space["a"], bshape=(env.num,)),
            "b": {
                "c": types_np.zeros(env.ac_space["b"]["c"], bshape=(env.num,)),
                "d": types_np.zeros(env.ac_space["b"]["d"], bshape=(env.num,)),
            },
        }
        action["a"][1] = 1
        action["b"]["c"][1] = 1
        action["b"]["d"][1] = 1
        num_acs = 10
        for _ in range(num_acs):
            env.act(action)
        files = sorted(glob(os.path.join(tmpdir, "*.pickle")))
        print(files)
        assert len(files) == (num_acs // ep_len1) + (num_acs // ep_len2)

        with open(files[0], "rb") as f:
            loaded_traj = pickle.load(f)

        assert len(loaded_traj["ob"]["a"]) == ep_len1
        assert len(loaded_traj["ob"]["b"]["c"]) == ep_len1
        assert len(loaded_traj["ob"]["b"]["d"]) == ep_len1

        assert np.allclose(loaded_traj["ob"]["a"][0], obs["a"][0])
        assert np.allclose(loaded_traj["ob"]["b"]["c"][0], obs["b"]["c"][0])
        assert np.allclose(loaded_traj["ob"]["b"]["d"][0], obs["b"]["d"][0])

        assert np.allclose(loaded_traj["act"]["a"][0], action["a"][0])
        assert np.allclose(loaded_traj["act"]["b"]["c"][0], action["b"]["c"][0])
        assert np.allclose(loaded_traj["act"]["b"]["d"][0], action["b"]["d"][0])

        with open(files[1], "rb") as f:
            loaded_traj = pickle.load(f)

        assert len(loaded_traj["ob"]["a"]) == ep_len2
        assert len(loaded_traj["ob"]["b"]["c"]) == ep_len2
        assert len(loaded_traj["ob"]["b"]["d"]) == ep_len2

        assert np.allclose(loaded_traj["ob"]["a"][0], obs["a"][1])
        assert np.allclose(loaded_traj["ob"]["b"]["c"][0], obs["b"]["c"][1])
        assert np.allclose(loaded_traj["ob"]["b"]["d"][0], obs["b"]["d"][1])

        assert np.allclose(loaded_traj["act"]["a"][0], action["a"][1])
        assert np.allclose(loaded_traj["act"]["b"]["c"][0], action["b"]["c"][1])
        assert np.allclose(loaded_traj["act"]["b"]["d"][0], action["b"]["d"][1])


if __name__ == "__main__":
    test_recorder()
    test_recorder_dict()
