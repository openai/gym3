import os
from typing import Any, Mapping, Optional

import imageio
import numpy as np

from gym3.env import Env
from gym3.internal.renderer import Renderer
from gym3.wrapper import Wrapper


class VideoRecorderWrapper(Wrapper):
    """
    Record observations of each episode from an environment to a video file

    Subclasses may want to override `_process_frame`

    :param env: environment to record from
    :param directory: directory to save videos to, will be created if it does not exist
    :param env_index: the index of the environment to record
    :param ob_key: by default the observation is recorded for the video, if the observation is a dictionary,
            you can specify which key to record using this argument
    :param info_key: if the frame you want to record is in the environment info dictionary, specify the key here, e.g. "rgb"
    :param prefix: filename prefix to use when creating videos
    :param fps: fps to give to encoder, this depends on your environment and the resulting
            video will play back too quickly or too slowly depending on this value
    :param writer_kwargs: extra arguments to supply to the imageio writer
    :param render: if set to True, also show the current frame being recorded in a window
    """

    def __init__(
        self,
        env: Env,
        directory: str,
        env_index: int = 0,
        ob_key: Optional[str] = None,
        info_key: Optional[str] = None,
        prefix: str = "",
        fps: int = 15,
        writer_kwargs: Optional[Mapping[str, Any]] = None,
        render=False,
    ) -> None:
        super().__init__(env=env)
        if info_key is not None:
            assert ob_key is None, "can't specify both info_key and ob_key"
        self._prefix = prefix
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)
        self._ob_key = ob_key
        self._info_key = info_key
        self._env_index = env_index
        self._episode_count = 0
        self._writer = None
        if writer_kwargs is None:
            writer_kwargs = {"output_params": ["-f", "mp4"]}
        self._writer_kwargs = writer_kwargs
        self._fps = fps
        self.videopath = None
        self._first_step = True
        self._renderer = Renderer(width=768, height=768) if render else None

    def _restart_recording(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self.videopath = os.path.join(
            self._directory, f"{self._prefix}{self._episode_count:05d}.mp4"
        )
        self._writer = imageio.get_writer(
            self.videopath, format="ffmpeg", fps=self._fps, **self._writer_kwargs
        )

    def _append_observation(self) -> None:
        _, ob, _ = self.observe()
        if self._info_key is None:
            if self._ob_key is not None:
                ob = ob[self._ob_key]
            img = ob[self._env_index]
        else:
            info = self.get_info()
            img = info[self._env_index].get(self._info_key)
            # the first info for a converted environment may be empty
            if self._first_step and img is None:
                return
        frame = self._process_frame(img.astype(np.uint8))
        self._writer.append_data(frame)
        if self._renderer is not None:
            self._renderer.start()
            self._renderer.draw_bitmap(
                0, 0, self._renderer.width, self._renderer.height, image=frame
            )
            self._renderer.finish()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def act(self, ac: Any) -> None:
        if self._first_step:
            # first action of the episode, get the existing observation before
            # taking an action
            self._restart_recording()
            self._append_observation()

        super().act(ac)
        self._first_step = False
        _, _, first = self.observe()
        if first[self._env_index]:
            self._episode_count += 1
            self._first_step = True
            self._writer.close()
            self._writer = None
        else:
            self._append_observation()
