import os
import pickle
from typing import Any

import numpy as np

from gym3.env import Env
from gym3.types import multimap
from gym3.types_np import concat, zeros
from gym3.wrapper import Wrapper


class TrajectoryRecorderWrapper(Wrapper):
    """
    Record a trajectory of each episode from an environment.

    Each saved file contains a single trajectory in pickle format, represented by a dictionary of lists of the same length as
    the trajectory. The dictionary keys are as follows:
    - "ob": list of observations
    - "act": list of actions. act[i] is the action taken after seeing ob[i]
    - "reward": list of rewards. reward[i] is the reward caused by taking act[i]
    - "info": list of metadata not observed by the agent. info[i] corresponds to the same timestep as ob[i]

    You can load a trajectory file like so:

        import pickle
        with open(filename, "rb") as f:
            trajectory = pickle.load(f)

    :param env: gym3 environment to record
    :param directory: directory to save trajectories to
    :param filename_prefix: use this prefix for the filenames of trajectories that are saved
    """

    def __init__(self, env: Env, directory: str, filename_prefix: str = "") -> None:
        super().__init__(env=env)
        self._prefix = filename_prefix
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)
        self._episode_count = 0
        self._trajectories = None
        self._ob_actual_dtype = None
        self._ac_actual_dtype = None

    def _new_trajectory_dict(self):
        assert self._ob_actual_dtype is not None, (
            "Not supposed to happen; self._ob_actual_dtype should have been set"
            " in the first act() call before _new_trajectory_dict is called"
        )
        traj_dict = dict(
            reward=list(),
            ob=zeros(self.env.ob_space, (0,)),
            info=list(),
            act=zeros(self.env.ac_space, (0,)),
        )
        traj_dict["ob"] = multimap(
            lambda arr, my_dtype: arr.astype(my_dtype),
            traj_dict["ob"],
            self._ob_actual_dtype,
        )
        traj_dict["act"] = multimap(
            lambda arr, my_dtype: arr.astype(my_dtype),
            traj_dict["act"],
            self._ac_actual_dtype,
        )
        return traj_dict

    def _write_and_reset_trajectory(self, idx) -> None:
        filepath = os.path.join(
            self._directory, f"{self._prefix}{self._episode_count:09d}.pickle"
        )
        with open(filepath, "wb") as f:
            self._trajectories[idx]['reward'] = np.array(self._trajectories[idx]['reward'])
            pickle.dump(self._trajectories[idx], f)
        self._trajectories[idx] = self._new_trajectory_dict()
        self._episode_count += 1

    def act(self, ac: Any) -> None:
        _, ob, _ = self.observe()
        info = self.get_info()

        # We have to wait for the first call to act() to initialize the _trajectories list, because
        # sometimes the environment returns observations with dtypes that do not match self.env.ob_space.
        if self._trajectories is None:
            self._ob_actual_dtype = multimap(lambda x: x.dtype, ob)
            self._ac_actual_dtype = multimap(lambda x: x.dtype, ac)
            self._trajectories = [
                self._new_trajectory_dict() for _ in range(self.env.num)
            ]

        for i in range(self.env.num):
            # With non-dict spaces, the `ob` and/or `ac` is a numpy array of shape [batch, obs_shape...] so separating
            # each trajectory into its own structure was relatively simple.
            # Take ob[i] then append it to self._trajectories[i]['ob'].
            #
            # With dict spaces, the returned ob becomes a nested dict
            # {
            #     'obs_key1': [batch, obs1_shape...],
            #     'obs_key2': [batch, obs2_shape...]
            # }
            # So to separate each trajectory, we have to take ob['obs_key1'][i] then append it to
            # self._trajectories[i]['ob']['obs_key1']
            self._trajectories[i]["ob"] = concat(
                [self._trajectories[i]["ob"], multimap(lambda x: x[i : i + 1], ob)],
                axis=0,
            )
            self._trajectories[i]["act"] = concat(
                [self._trajectories[i]["act"], multimap(lambda x: x[i : i + 1], ac)],
                axis=0,
            )
            self._trajectories[i]["info"].append(info[i])

        super().act(ac)

        reward, _, first = self.observe()
        for i in range(self.env.num):
            self._trajectories[i]["reward"].append(reward[i])

        # For each completed trajectory, write it out
        for i in range(self.env.num):
            if first[i]:
                self._write_and_reset_trajectory(i)
