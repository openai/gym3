"""
Python code to interact with an environment using the libenv C interface

See libenv.h for more details and libenv_fixedseq.c for a simple example environment
"""

import collections
import copy
import os
import platform
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from cffi import FFI

from gym3 import types
from gym3.env import Env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBENV_VERSION = 1  # also appears in libenv.h

FFI_CACHE_LOCK = threading.Lock()
FFI_CACHE = {}


def get_header_dir():
    return SCRIPT_DIR


def _load_libenv_cdef():
    libenv_cdef = ""
    with open(os.path.join(get_header_dir(), "libenv.h")) as f:
        inside_cdef = False
        for line in f:
            if line.startswith("// BEGIN_CDEF"):
                inside_cdef = True
            elif line.startswith("// END_CDEF"):
                inside_cdef = False
            elif line.startswith("#if") or line.startswith("#endif"):
                continue

            if inside_cdef:
                line = line.replace("LIBENV_API", "")
                libenv_cdef += line
    return libenv_cdef


Spec = collections.namedtuple("Spec", ["name", "shape", "dtype"])


class CEnv(Env):
    """
    An environment instance backed by a shared library implementing the libenv interface

    :param num: number of environments to create
    :param lib_dir: a folder containing either lib{name}.so (Linux), lib{name}.dylib (Mac), or {name}.dll (Windows)
    :param lib_name: name of the library (minus the lib part)
    :param c_func_defs: list of cdefs that are passed to FFI in order to define custom functions that can then be called with env.call_func()
    :param options: options to pass to the libenv_make() call for this environment
    :param reuse_arrays: reduce allocations by using the same numpy arrays for each reset(), step(), and render() call
    """

    def __init__(
        self,
        num: int,
        lib_dir: str,
        lib_name: str = "env",
        c_func_defs: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        reuse_arrays: bool = False,
    ) -> None:
        self._reuse_arrays = reuse_arrays
        self.num = num

        if options is None:
            options = {}
        options = copy.deepcopy(options)

        if platform.system() == "Linux":
            lib_filename = f"lib{lib_name}.so"
        elif platform.system() == "Darwin":
            lib_filename = f"lib{lib_name}.dylib"
        elif platform.system() == "Windows":
            lib_filename = f"{lib_name}.dll"
        else:
            raise Exception(f"unrecognized platform {platform.system()}")

        if c_func_defs is None:
            c_func_defs = []

        with FFI_CACHE_LOCK:
            key = tuple(c_func_defs)
            if key not in FFI_CACHE:
                ffi = FFI()
                ffi.cdef(_load_libenv_cdef())

                for cdef in c_func_defs:
                    ffi.cdef(cdef)
                FFI_CACHE[key] = ffi
            self._ffi = FFI_CACHE[key]

        self._lib_path = os.path.join(lib_dir, lib_filename)
        assert os.path.exists(self._lib_path), f"lib not found at {self._lib_path}"
        # unclear if this is necessary, but nice to not have symbols conflict if possible
        dlopen_flags = (
            self._ffi.RTLD_NOW | self._ffi.RTLD_LOCAL  # pylint: disable=no-member
        )
        if platform.system() == "Linux":
            dlopen_flags |= self._ffi.RTLD_DEEPBIND  # pylint: disable=no-member
        self._c_lib = self._ffi.dlopen(name=self._lib_path, flags=dlopen_flags)
        # dlclose will be called automatically when the library goes out of scope
        # https://cffi.readthedocs.io/en/latest/cdef.html#ffi-dlopen-loading-libraries-in-abi-mode
        # on mac os x, the library may not always be unloaded when you expect
        # https://developer.apple.com/videos/play/wwdc2017/413/?time=1776
        # loading/unloading the library all the time can be slow
        # it may be useful for the user to keep a reference to an environment (and thus the c_lib object)
        # to avoid this happening

        version = self._c_lib.libenv_version()
        assert (
            version == LIBENV_VERSION
        ), f"libenv version mismatch, got {version} but expected {LIBENV_VERSION}"

        self._options = options

        c_options, self._options_keepalives = self._convert_options(
            self._ffi, self._c_lib, options
        )

        self._c_env = self._c_lib.libenv_make(num, c_options[0])

        ob_space, ob_specs = self._get_space(self._c_lib.LIBENV_SPACE_OBSERVATION)
        ac_space, ac_specs = self._get_space(self._c_lib.LIBENV_SPACE_ACTION)
        self._info_space, info_specs = self._get_space(self._c_lib.LIBENV_SPACE_INFO)

        # allocate buffers, the buffers are assigned to the current object to keep them alive
        self._ob, self._c_ob_buffers = self._allocate_specs(self.num, ob_specs)

        self._ac, self._c_ac_buffers = self._allocate_specs(self.num, ac_specs)

        self._info, self._c_info_buffers = self._allocate_specs(self.num, info_specs)

        self._rew, self._c_rew_buffer = self._allocate_array(
            self.num, np.dtype("float32")
        )
        self._first, self._c_first_buffer = self._allocate_array(
            self.num, np.dtype("bool")
        )
        assert np.dtype("bool").itemsize == 1

        self._c_buffers = self._ffi.new("struct libenv_buffers *")
        # cast the pointer to the buffer to avoid a warning from cffi
        self._c_buffers.rew = self._ffi.cast(
            self._ffi.typeof(self._c_buffers.rew).cname, self._c_rew_buffer
        )
        self._c_buffers.ob = self._c_ob_buffers
        self._c_buffers.first = self._ffi.cast(
            self._ffi.typeof(self._c_buffers.first).cname, self._c_first_buffer
        )
        self._c_buffers.ac = self._c_ac_buffers
        self._c_buffers.info = self._c_info_buffers

        self._c_lib.libenv_set_buffers(self._c_env, self._c_buffers)

        self.closed = False
        super().__init__(ob_space=ob_space, ac_space=ac_space, num=num)

    def __repr__(self) -> str:
        return f"<CEnv lib_path={self._lib_path} options={self._options}>"

    def _numpy_aligned(
        self, shape: Tuple[int], dtype: np.dtype, align: int = 64
    ) -> np.ndarray:
        """
        Allocate an aligned numpy array, based on https://github.com/numpy/numpy/issues/5312#issuecomment-299533915
        """
        n_bytes = np.prod(shape) * dtype.itemsize
        arr = np.zeros(n_bytes + (align - 1), dtype=np.uint8)
        data_align = arr.ctypes.data % align
        offset = 0 if data_align == 0 else (align - data_align)
        view = arr[offset : offset + n_bytes].view(dtype)
        return view.reshape(shape)

    def _allocate_specs(
        self, num: int, specs: Sequence[Spec]
    ) -> Tuple[Dict[str, np.ndarray], Any]:
        """
        Allocate arrays for a space, returns a dict of numpy arrays along with an array of void * pointers
        """
        result = {}
        length = len(specs) * num
        buffers = self._ffi.new(f"void *[{length}]")
        for space_idx, spec in enumerate(specs):
            actual_shape = (num,) + spec.shape
            arr = self._numpy_aligned(shape=actual_shape, dtype=spec.dtype)
            result[spec.name] = arr
            for env_idx in range(num):
                buffers[space_idx * num + env_idx] = self._ffi.from_buffer(
                    arr.data[
                        env_idx : env_idx + 1
                    ]  # this is just to get the address, the length is not used
                )
        return result, buffers

    def _allocate_array(self, num: int, dtype: np.dtype) -> Tuple[np.ndarray, Any]:
        arr = self._numpy_aligned(shape=(num,), dtype=dtype)
        return arr, self._ffi.from_buffer(arr.data)

    @staticmethod
    def _convert_options(ffi: Any, c_lib: Any, options: Dict) -> Any:
        """
        Convert a dictionary to libenv_options
        """
        # add variables to here to keep them alive after this function returns
        keepalives = []
        c_options = ffi.new("struct libenv_options *")
        c_option_array = ffi.new("struct libenv_option[%d]" % len(options))

        for i, (k, v) in enumerate(options.items()):
            name = str(k).encode("utf8")
            assert (
                len(name) < c_lib.LIBENV_MAX_NAME_LEN - 1
            ), "length of options key is too long"
            if isinstance(v, bytes):
                c_data = ffi.new("char[]", v)
                dtype = c_lib.LIBENV_DTYPE_UINT8
                count = len(v)
            elif isinstance(v, str):
                c_data = ffi.new("char[]", v.encode("utf8"))
                dtype = c_lib.LIBENV_DTYPE_UINT8
                count = len(v)
            elif isinstance(v, bool):
                c_data = ffi.new("uint8_t*", v)
                dtype = c_lib.LIBENV_DTYPE_UINT8
                count = 1
            elif isinstance(v, int):
                assert -2 ** 31 < v < 2 ** 31
                c_data = ffi.new("int32_t*", v)
                dtype = c_lib.LIBENV_DTYPE_INT32
                count = 1
            elif isinstance(v, float):
                c_data = ffi.new("float*", v)
                dtype = c_lib.LIBENV_DTYPE_FLOAT32
                count = 1
            elif isinstance(v, np.ndarray):
                c_data = ffi.new("char[]", v.tobytes())
                if v.dtype == np.dtype("uint8"):
                    dtype = c_lib.LIBENV_DTYPE_UINT8
                elif v.dtype == np.dtype("int32"):
                    dtype = c_lib.LIBENV_DTYPE_INT32
                elif v.dtype == np.dtype("float32"):
                    dtype = c_lib.LIBENV_DTYPE_FLOAT32
                else:
                    assert False, f"unsupported type {v.dtype}"
                count = v.size
            else:
                assert False, f"unsupported value {v} for option {k}"

            c_option_array[i].name = name
            c_option_array[i].dtype = dtype
            c_option_array[i].count = count
            c_option_array[i].data = c_data
            keepalives.append(c_data)

        keepalives.append(c_option_array)
        c_options.items = c_option_array
        c_options.count = len(options)
        return c_options, keepalives

    def _maybe_copy_ndarray(self, obj: np.ndarray) -> np.ndarray:
        """
        Copy a single numpy array if reuse_arrays is False,
        otherwise just return the object
        """
        if self._reuse_arrays:
            return obj
        else:
            return obj.copy()

    def _maybe_copy_dict(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy a list of dicts of numpy arrays if reuse_arrays is False,
        otherwise just return the object
        """
        if self._reuse_arrays:
            return obj
        else:
            result = {}
            for name, arr in obj.items():
                result[name] = arr.copy()
            return result

    def _get_space(self, c_name: Any) -> Tuple[types.DictType, List[Spec]]:
        """
        Get a c space and convert to a gym space
        """
        count = self._c_lib.libenv_get_tensortypes(self._c_env, c_name, self._ffi.NULL)
        if count == 0:
            return types.DictType(), []

        c_tensortypes = self._ffi.new("struct libenv_tensortype[%d]" % count)
        self._c_lib.libenv_get_tensortypes(self._c_env, c_name, c_tensortypes)

        # convert to gym3 types
        name_to_tensortype = {}
        specs = []
        for i in range(count):
            c_tt = c_tensortypes[i]

            name = self._ffi.string(c_tt.name).decode("utf8")
            shape = tuple(c_tt.shape[j] for j in range(c_tt.ndim))

            if c_tt.scalar_type == self._c_lib.LIBENV_SCALAR_TYPE_REAL:
                if c_tt.dtype == self._c_lib.LIBENV_DTYPE_FLOAT32:
                    dtype = np.dtype("float32")
                    low = c_tt.low.float32
                    high = c_tt.high.float32
                else:
                    assert False, "unrecognized dtype for real"
                tensortype = types.TensorType(eltype=types.Real(), shape=shape)
            elif c_tt.scalar_type == self._c_lib.LIBENV_SCALAR_TYPE_DISCRETE:
                if c_tt.dtype == self._c_lib.LIBENV_DTYPE_UINT8:
                    dtype = np.dtype("uint8")
                    low = c_tt.low.uint8
                    high = c_tt.high.uint8
                elif c_tt.dtype == self._c_lib.LIBENV_DTYPE_INT32:
                    dtype = np.dtype("int32")
                    low = c_tt.low.int32
                    high = c_tt.high.int32
                else:
                    assert False, "unrecognized dtype for discrete"
                assert low == 0 and high >= 0, "discrete low/high bounds are incorrect"
                tensortype = types.TensorType(
                    eltype=types.Discrete(n=high + 1, dtype_name=dtype.name),
                    shape=shape,
                )
            else:
                assert False, "unknown space type"
            name_to_tensortype[name] = tensortype
            specs.append(Spec(name=name, shape=tensortype.shape, dtype=dtype))
        return types.DictType(**name_to_tensortype), specs

    def observe(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        self._c_lib.libenv_observe(self._c_env)
        return (
            self._maybe_copy_ndarray(self._rew),
            self._maybe_copy_dict(self._ob),
            self._maybe_copy_ndarray(self._first),
        )

    def act(self, ac: Dict[str, np.ndarray]) -> None:
        for key, value in ac.items():
            assert (
                self._ac[key].shape == value.shape
            ), f"action shape did not match expected={self._ac[key].shape} actual={value.shape}"
            assert (
                self._ac[key].dtype == value.dtype
            ), f"action dtype did not match expected={self._ac[key].dtype} actual={value.dtype}"
            self._ac[key][:] = value
        self._c_lib.libenv_act(self._c_env)

    def get_info(self) -> List[Dict[str, Any]]:
        self._c_lib.libenv_observe(self._c_env)
        infos = [{} for _ in range(self.num)]
        info = self._maybe_copy_dict(self._info)
        for key, values in info.items():
            for env_idx in range(self.num):
                infos[env_idx][key] = values[env_idx]
        return infos

    def close(self) -> None:
        """Close the environment and free any resources associated with it"""
        if not hasattr(self, "closed") or self.closed:
            return

        self.closed = True
        self._c_lib.libenv_close(self._c_env)
        self._c_lib = None
        self._ffi = None
        self._options_keepalives = None

    def call_c_func(self, name: str, *args: Any) -> Any:
        """
        Call a function of the libenv declared in c_func_defs

        The function's first argument must be an environment handle, this will be added automatically
        when the function is called.
        """
        return getattr(self._c_lib, name)(self._c_env, *args)

    def __del__(self):
        self.close()
