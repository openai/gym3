import concurrent.futures
from typing import Any, Dict, List, Tuple

from gym3.env import Env
from gym3.wrapper import Wrapper


class AsynchronousWrapper(Wrapper):
    """
    For environments with a synchronous act() function, run act() asynchronously on a
    separate thread.

    :param env: environment to wrap
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._futures = []

    def _wait(self):
        if len(self._futures) > 0:
            # wait for all pending act calls to complete
            for future in self._futures:
                # this will re-raise any exceptions from the worker thread
                future.result()
            self._futures = []

    def observe(self) -> Tuple[Any, Any, Any]:
        self._wait()
        return self.env.observe()

    def get_info(self) -> List[Dict]:
        self._wait()
        return self.env.get_info()

    def act(self, ac: Any) -> None:
        future = self._executor.submit(self.env.act, ac)
        self._futures.append(future)
