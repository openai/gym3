import contextlib
import os
import platform
import subprocess as sp
import tempfile

import pytest
import numpy as np

from gym3.libenv import CEnv
from gym3.testing import AssertSpacesWrapper, FixedSequenceEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def chdir(newdir):
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


class CFixedSequenceEnv(CEnv):
    def __init__(self, num, n_actions, episode_len, sequence):
        options = dict(n_actions=n_actions, episode_len=episode_len, sequence=sequence)
        super().__init__(num=num, lib_dir=".", options=options)
        self.ob_space = self.ob_space["ignore"]
        self.ac_space = self.ac_space["action"]

    def observe(self):
        rew, ob, first = super().observe()
        return rew, ob["ignore"], first

    def act(self, ac):
        super().act({"action": ac})

 
@pytest.mark.skipif(platform.system() == "Windows", reason="MSVC build not supported yet")
def test_libenv_fixedseq():
    ref_env = FixedSequenceEnv(num=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        with chdir(tmpdir):
            ext = "dylib" if platform.system() == "Darwin" else "so"
            sp.run(
                [
                    "gcc",
                    "-ggdb",
                    "-shared",
                    "-fpic",
                    os.path.join(SCRIPT_DIR, "libenv_fixedseq.c"),
                    "-o",
                    f"libenv.{ext}",
                ],
                check=True,
            )

            env = CFixedSequenceEnv(
                num=ref_env.num,
                n_actions=ref_env.actions.shape[0],
                episode_len=ref_env.episode_len,
                sequence=ref_env.sequence.astype(np.uint8),
            )
            env = AssertSpacesWrapper(env)

            for i in range(1000):
                rew, ob, first = env.observe()
                ref_rew, ref_ob, ref_first = ref_env.observe()
                assert np.array_equal(ref_rew, rew)
                assert np.array_equal(ref_ob, ob)
                assert np.array_equal(ref_first, first)
                act = np.array([1, 1], dtype=np.uint8)
                env.act(act)
                ref_env.act(act)


if __name__ == "__main__":
    test_libenv_fixedseq()
