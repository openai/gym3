import os
import tempfile
from glob import glob

import imageio
import numpy as np

from gym3 import types, types_np
from gym3.testing import IdentityEnv
from gym3.video_recorder import VideoRecorderWrapper


def test_recorder():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = IdentityEnv(
            space=types.TensorType(eltype=types.Discrete(256), shape=(64, 64, 3))
        )
        writer_kwargs = {
            "codec": "libx264rgb",
            "pixelformat": "bgr24",
            "output_params": ["-crf", "0"],
        }
        env = VideoRecorderWrapper(
            env=env, directory=tmpdir, env_index=0, writer_kwargs=writer_kwargs
        )
        _, obs, _ = env.observe()
        for _ in range(2):
            env.act(types_np.zeros(env.ac_space, bshape=(env.num,)))
        video_files = sorted(glob(os.path.join(tmpdir, "*.mp4")))
        assert len(video_files) > 0
        with imageio.get_reader(video_files[0]) as r:
            for im in r:
                assert np.allclose(im, obs[0])
                break


if __name__ == "__main__":
    test_recorder()
