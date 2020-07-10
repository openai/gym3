import time
from typing import Any, Optional

import numpy as np

from gym3 import types_np
from gym3.env import Env
from gym3.internal.renderer import Renderer
from gym3.wrapper import Wrapper

HELP_TEXT = """\
p: (p)ause
s: (s)ingle step (when paused)
f: toggle (f)ast mode
o: toggle (o)verlay"""


class ViewerWrapper(Wrapper):
    """
    Displays images (from observations or from the info dict) to a human in a desktop window

    :param env: environment to record from
    :param env_index: the index of the environment to record
    :param ob_key: by default the observation is recorded for the video, if the observation is a dictionary,
            you can specify which key to record using this argument
    :param info_key: if the frame you want to record is in the environment info dictionary, specify the key here, e.g. "rgb"
    :param width: width of the window in pixels
    :param height: height of the window in pixels
    :param tps: timesteps per second of the environment
    """

    def __init__(
        self,
        env: Env,
        env_index: int = 0,
        ob_key: Optional[str] = None,
        info_key: Optional[str] = None,
        width: int = 768,
        height: int = 768,
        tps: int = 15,
    ) -> None:
        super().__init__(env=env)
        self._ob_key = ob_key
        self._info_key = info_key
        self._env_index = env_index
        self._sec_per_timestep = 1 / tps
        self._renderer = Renderer(width=width, height=height)
        self._last_frame_time = self._renderer.get_time()
        self._paused = False
        self._overlay_enabled = True
        self._fast_mode = False

    def _get_image(self) -> None:
        _, ob, _ = self.observe()
        if self._info_key is None:
            if self._ob_key is not None:
                ob = ob[self._ob_key]
            return ob[self._env_index]
        else:
            info = self.get_info()
            return info[self._env_index][self._info_key]

    def _render_image(self, image: np.array) -> None:
        self._renderer.draw_bitmap(
            0, 0, self._renderer.width, self._renderer.height, image=image
        )
        if self._paused:
            self._renderer.draw_text(
                self._renderer.width // 2,
                self._renderer.height // 6,
                text="(PAUSED)",
                centered=True,
                bg_alpha=0.5,
            )
        if self._fast_mode:
            self._renderer.draw_text(
                self._renderer.width // 2,
                self._renderer.height - self._renderer.height // 6,
                text="(FAST MODE)",
                centered=True,
                bg_alpha=0.5,
            )
        if self._overlay_enabled:
            self._renderer.draw_text(10, 10, text=HELP_TEXT, bg_alpha=0.5, size_px=16)
        self._renderer.finish()

        # sleep to maintain framerate
        now = self._renderer.get_time()
        if (now - self._last_frame_time) < self._sec_per_timestep:
            sleep_time = self._sec_per_timestep - (now - self._last_frame_time)
            if not self._fast_mode:
                time.sleep(sleep_time)
        self._last_frame_time = self._renderer.get_time()

    def act(self, ac: Any) -> None:
        super().act(ac)
        image = self._get_image()
        assert (
            len(image.shape) == 3 and image.shape[-1] == 3
        ), "expected (H, W, C) RGB image with C = 3"
        assert image.dtype.name == "uint8", "expected uint8 image"
        while self._renderer.is_open:
            keys_clicked, _ = self._renderer.start()

            # render all the time in slow mode, only render once per second in fast mode
            if (
                not self._fast_mode
                or self._renderer.get_time() - self._last_frame_time > 1
            ):
                self._render_image(image)

            if "F" in keys_clicked:
                self._fast_mode = not self._fast_mode

            if "P" in keys_clicked:
                self._paused = not self._paused

            if "O" in keys_clicked:
                self._overlay_enabled = not self._overlay_enabled

            if not self._paused or "S" in keys_clicked:
                break


def main():
    import procgen

    env = procgen.ProcgenGym3Env(num=1, env_name="coinrun", render_mode="rgb_array")
    env = ViewerWrapper(env=env, info_key="rgb")
    start = time.time()
    for i in range(10000):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        print("step", i, i / (time.time() - start))


if __name__ == "__main__":
    main()
