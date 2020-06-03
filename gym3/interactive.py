"""
Class that lets humans interact with gym3 environments using a keyboard
"""

import copy
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from gym3 import types_np
from gym3.env import Env
from gym3.internal.renderer import Renderer

SECONDS_TO_DISPLAY_DONE_INFO = 3


class Interactive:
    """
    Interact with gym3 environments using the keyboard

    Each environment must provide a function that maps keyboard key names to actions

    Subclasses may want to override _format_info(), _get_image() or _draw_step()

    To see an example of this script in action, run:

        pip install procgen
        python -m gym3.interactive

    :param env: the gym3.Env instance to use
    :param keys_to_act: function that takes a list of key names and returns an action for the environment (or None for no action)
                if this is not specified, the Interactive class will use callmethod to call "keys_to_act" on the environment
                which should take List[List[str]] and return List[np.ndarray] (the outer list is required for callmethod)
                key names are the GLFW key names, without the "GLFW_KEY_" prefix: https://www.glfw.org/docs/3.3/group__keys.html
    :param synchronous: if set to True, only takes steps when keys_to_act returns a non-None value
    :param ob_key: if set, the key in the observation dict that contains the RGB data to display,
                if set to None (the default), uses the observation itself as the RGB data
    :param info_key: if set, the key in the info dict that contains the RGB data to display, overrides ob_key
    :param width: width of the window to create in pixels
    :param height: height of the window to create in pixels
    :param tps: timesteps per second to limit the simulation speed
    """

    def __init__(
        self,
        env: Env,
        keys_to_act: Optional[Callable[[Sequence[str]], Optional[np.ndarray]]] = None,
        synchronous: bool = False,
        ob_key: Optional[str] = None,
        info_key: Optional[str] = None,
        width: int = 768,
        height: int = 768,
        tps: int = 15,
    ) -> None:
        super().__init__()
        self._ob_key = ob_key
        self._info_key = info_key
        assert env.num == 1, "interactive only supports environments with num=1"
        self._env = env
        self._tps = tps
        self._sec_per_timestep = 1 / tps
        self._renderer = Renderer(width=width, height=height)
        if keys_to_act is None:
            keys_to_act = lambda keys: env.callmethod("keys_to_act", [keys])[0]
        self._keys_to_act = keys_to_act
        self._synchronous = synchronous
        self._display_info_seconds_remaining = 0

        self._steps = 0
        self._episode_steps = 0
        self._episode_return = 0
        self._prev_episode_return = 0
        self._last_ob = None
        self._last_ac = None
        self._last_info = {}
        self._last_rew = None
        self._overlay_enabled = True

        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

    def run(self) -> None:
        """
        Display a window to the user and loop until the window is closed
        by the user.
        """
        prev_time = self._renderer.get_time()
        self._renderer.start()
        self._draw()
        self._renderer.finish()

        while True:
            now = self._renderer.get_time()
            dt = now - prev_time
            prev_time = now
            if dt < self._sec_per_timestep:
                sleep_time = self._sec_per_timestep - dt
                time.sleep(sleep_time)

            keys_clicked, keys_pressed = self._renderer.start()
            if "O" in keys_clicked:
                self._overlay_enabled = not self._overlay_enabled
            self._update(dt, keys_clicked, keys_pressed)
            self._draw()
            self._renderer.finish()
            if not self._renderer.is_open:
                break

    def _get_image(self) -> Optional[np.ndarray]:
        """
        Get the image that we should display to the user for the current step
        """
        _, ob, _ = self._env.observe()
        if self._info_key is None:
            if self._ob_key is not None:
                ob = ob[self._ob_key]
            return ob[0]
        else:
            info = self._env.get_info()
            return info[0].get(self._info_key)

    def _format_info(self) -> str:
        """
        Format the info for the current step into a string
        """
        info_rows = []
        for k, v in sorted(self._last_info.items()):
            info_rows.append(f"{k}: {v}")
        return "\n".join(info_rows)

    def _draw_step(self) -> None:
        """
        Draw the image for the current step from the environment to the screen, as well as any overlays
        """
        image = self._get_image()
        self._renderer.draw_bitmap(
            0, 0, self._renderer.width, self._renderer.height, image=image
        )
        if self._overlay_enabled:
            text = self._format_info()
            if len(text) > 0:
                self._renderer.draw_text(
                    10,
                    10,
                    text=self._format_info() + "\npress o to toggle overlay",
                    bg_alpha=0.5,
                    size_px=16,
                )

    def _draw(self) -> None:
        if self._display_info_seconds_remaining > 0:
            text = "=== episode complete ===\n\n" + self._format_info()
            self._renderer.draw_text(
                self._renderer.width // 2,
                self._renderer.height // 2,
                text=text,
                centered=True,
            )
        else:
            image = self._get_image()
            if image is None:
                self._renderer.draw_text(
                    self._renderer.width // 2,
                    self._renderer.height // 2,
                    text="(missing image)",
                    centered=True,
                )
            else:
                assert (
                    len(image.shape) == 3 and image.shape[-1] == 3
                ), "expected (H, W, C) RGB image with C = 3"
                assert image.dtype.name == "uint8", "expected uint8 image"
                self._draw_step()

    def _act(self, ac: Any) -> None:
        self._env.act(ac)
        batch_rew, batch_obs, batch_first = self._env.observe()
        self._last_rew = batch_rew[0]
        self._last_ob = batch_obs
        self._last_ac = ac
        first = batch_first[0]
        info = copy.copy(self._env.get_info()[0])
        for k in list(info.keys()):
            if isinstance(info[k], np.ndarray):
                del info[k]

        self._episode_return += self._last_rew
        self._steps += 1
        self._episode_steps += 1
        self._last_info = dict(
            episode_steps=self._episode_steps,
            episode_return=self._episode_return,
            **info,
        )
        np.set_printoptions(precision=2)
        return first

    def _update(self, dt, keys_clicked, keys_pressed):
        # if we're displaying done info, don't advance the simulation
        if self._display_info_seconds_remaining > 0:
            self._display_info_seconds_remaining -= dt
            return

        first = False

        if self._synchronous:
            keys = keys_clicked
            act = self._keys_to_act(keys)

            if act is not None:
                first = self._act(act)
                print(
                    "first={} steps={} episode_steps={} rew={} episode_return={}".format(
                        int(first),  # shoter than printing True/False
                        self._steps,
                        self._episode_steps,
                        self._last_rew,
                        self._episode_return,
                    )
                )
        else:
            # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
            # if rendering is slow
            max_dt = self._max_sim_frames_per_update * self._sec_per_timestep
            if dt > max_dt:
                dt = max_dt

            # catch up the simulation to the current time
            self._current_time += dt
            while self._sim_time < self._current_time or self._synchronous:
                self._sim_time += self._sec_per_timestep

                # assume that for async environments, we just want to repeat keys for as long as they are held
                keys = keys_pressed

                act = self._keys_to_act(keys)
                if act is None:
                    act = types_np.zeros(self._env.ac_space, bshape=(self._env.num,))

                first = self._act(act)
                if self._steps % self._tps == 0 or first:
                    episode_return_delta = (
                        self._episode_return - self._prev_episode_return
                    )
                    self._prev_episode_return = self._episode_return
                    print(
                        "first={} steps={} episode_steps={} episode_return_delta={} episode_return={}".format(
                            int(first),
                            self._steps,
                            self._episode_steps,
                            episode_return_delta,
                            self._episode_return,
                        )
                    )

        if first:
            print(f"final info={self._last_info}")
            self._episode_steps = 0
            self._episode_return = 0
            self._prev_episode_return = 0
            self._display_info_seconds_remaining = SECONDS_TO_DISPLAY_DONE_INFO


def main():
    from procgen import ProcgenGym3Env

    env = ProcgenGym3Env(num=1, env_name="coinrun", render_mode="rgb_array")
    ia = Interactive(env, info_key="rgb", width=768, height=768)
    ia.run()


if __name__ == "__main__":
    main()
