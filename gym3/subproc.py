import contextlib
import multiprocessing as mp
import os
import pickle
import threading
import traceback
from typing import Any, Dict, List, Sequence, Tuple

from gym3.env import Env

_clear_lock = threading.Lock()

CODECS = {"pickle": (pickle.dumps, pickle.loads)}

try:
    import cloudpickle
except ImportError:
    pass
else:
    CODECS["cloudpickle"] = (cloudpickle.dumps, cloudpickle.loads)


class SubprocError(Exception):
    pass


class SubprocEnv(Env):
    """
    Create an environment in a subprocess using the provided function.

    :param env_fn: function to call to create the gym environment, defaults to `gym.make` 
    :param env_kwargs: keyword arguments to pass to env_fn
    :param daemon: if set to False, don't create daemon processes, the parent process will block
        when exiting until all non-daemon child processes have exited
    """

    def __init__(self, env_fn, env_kwargs=None, daemon=True):
        if env_kwargs is None:
            env_kwargs = {}
        # tensorflow is not fork safe, and fork doesn't work on all platforms anyway
        self._ctx = mp.get_context("spawn")
        self._p2c, c2p = self._ctx.Pipe()

        # pickle cannot pickle functions, so fallback to cloudpickle if pickle fails
        last_err = None
        for codec, (encode, decode) in CODECS.items():
            try:
                env_fn_serialized = encode(env_fn)
                env_kwargs_serialized = encode(env_kwargs)
            except Exception as e:
                last_err = e
            else:
                break
        else:
            raise Exception(
                f"all attempted encoders failed, tried: {', '.join(CODECS.keys())}.  Last error was:\n  {last_err}.\n\nIf you are pickling a function defined inside of another function, try `pip install cloudpickle` to enable cloudpickle encoding"
            )

        self._child = self._ctx.Process(
            target=_worker,
            kwargs=dict(
                decode=decode,
                env_fn_serialized=env_fn_serialized,
                env_kwargs_serialized=env_kwargs_serialized,
                p2c=self._p2c,
                c2p=c2p,
            ),
            daemon=daemon,
        )
        # clear mpi vars to avoid issues with MPI_init being called in subprocesses
        with _clear_mpi_env_vars():
            self._child.start()
        # close child connection to avoid hangs when child exits unexpectedly
        c2p.close()
        result, err = self._communicate(self._p2c.recv)
        if err is not None:
            self.close()
            raise SubprocError("failed to create env in subprocess") from Exception(
                "exception in subprocess:\n\n" + err
            )
        ob_space, ac_space, num = result
        super().__init__(ob_space=ob_space, ac_space=ac_space, num=num)

    def observe(self) -> Tuple[Any, Any, Any]:
        return self._call_method_in_worker("observe")

    def get_info(self) -> List[Dict]:
        return self._call_method_in_worker("get_info")

    def act(self, ac: Any) -> None:
        self._call_method_in_worker_noreturn("act", ac=ac)

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return self._call_method_in_worker("callmethod", method, *args, **kwargs)

    def _communicate(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (EOFError, ConnectionResetError):
            self._child.join()
            raise SubprocError(
                f"child process exited with exit code {self._child.exitcode}"
            )

    def _call_method_in_worker(self, method, *args, **kwargs):
        self._communicate(
            self._p2c.send,
            dict(send_response=True, method=method, args=args, kwargs=kwargs),
        )
        result, err = self._communicate(self._p2c.recv)
        if err is not None:
            raise SubprocError from Exception("exception in subprocess:\n\n" + err)
        return result

    def _call_method_in_worker_noreturn(self, method, *args, **kwargs):
        self._communicate(
            self._p2c.send,
            dict(send_response=False, method=method, args=args, kwargs=kwargs),
        )

    def close(self):
        if hasattr(self, "_child") and self._child is not None:
            if self._child.is_alive():
                self._p2c.send(None)
                self._child.join()
                self._p2c.close()
            self._child = None
            self._p2c = None

    def __del__(self):
        self.close()


def _worker(p2c, c2p, decode, env_fn_serialized, env_kwargs_serialized):
    try:
        p2c.close()
        result = None
        err = None
        try:
            env_fn = decode(env_fn_serialized)
            env_kwargs = decode(env_kwargs_serialized)
            env = env_fn(**env_kwargs)
        except Exception as e:
            err = traceback.format_exc()
            c2p.send((result, err))
            return
        else:
            result = (env.ob_space, env.ac_space, env.num)
            c2p.send((result, err))

        while True:
            msg = c2p.recv()
            if msg is None:
                # this is sent to tell the child to exit
                return

            result = None
            try:
                fn = getattr(env, msg["method"])
                result = fn(*msg["args"], **msg["kwargs"])
            except Exception as e:
                err = traceback.format_exc()
            if msg["send_response"]:
                c2p.send((result, err))
            # if send_response is False but an error occurred, the an error will be sent the next time
            # send_response is True
    except KeyboardInterrupt:
        print("Subproc worker: got KeyboardInterrupt")


@contextlib.contextmanager
def _clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If we spawn a child process that also calls MPI_Init and
    has MPI environment variables defined, MPI will think that the child process is an MPI process just like the
    parent and do bad things such as hang or crash.

    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting
    multiprocessing Processes.
    """
    with _clear_lock:
        removed_environment = {}
        for k, v in list(os.environ.items()):
            for prefix in ["OMPI_", "PMI_"]:
                if k.startswith(prefix):
                    removed_environment[k] = v
                    del os.environ[k]
        try:
            yield
        finally:
            os.environ.update(removed_environment)
