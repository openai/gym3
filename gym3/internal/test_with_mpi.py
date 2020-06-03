import base64
import os
import pickle
import subprocess
import sys

from gym3.util import call_func


def run_test_with_mpi(fn_path, kwargs=None, nproc=2, timeout=30):
    if kwargs is None:
        kwargs = {}
    serialized_fn = base64.b64encode(pickle.dumps((fn_path, kwargs)))
    subprocess.check_call(
        [
            "mpiexec",
            "-n",
            str(nproc),
            sys.executable,
            "-m",
            "gym3.internal.test_with_mpi",
            serialized_fn,
        ],
        env=os.environ,
        timeout=timeout,
    )


if __name__ == "__main__":
    fn_path, kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    call_func(fn_path, **kwargs)
