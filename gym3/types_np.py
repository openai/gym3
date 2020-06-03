from functools import partial
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from gym3.types import Discrete, Real, TensorType, ValType, multimap


def concat(xs: Sequence[Any], axis: int = 0) -> Any:
    """
    Concatenate the (leaf) arrays from xs

    :param xs: list of trees with the same shape, where the leaf values are numpy arrays
    :param axis: axis to concatenate along
    """
    return multimap(lambda *xs: np.concatenate(xs, axis=axis), *xs)


def stack(xs: Sequence[Any], axis: int = 0) -> Any:
    """
    Stack the (leaf) arrays from xs

    :param xs: list of trees with the same shape, where the leaf values are numpy arrays
    :param axis: axis to stack along
    """
    return multimap(lambda *xs: np.stack(xs, axis=axis), *xs)


def split(x: Any, sections: Sequence[int]) -> Sequence[Any]:
    """
    Split the (leaf) arrays from the tree x

    Examples:

        split([1,2,3,4], [1,2,3,4]) => [[1], [2], [3], [4]]
        split([1,2,3,4], [1,3,4]) => [[1], [2, 3], [4]]

    :param x: a tree where the leaf values are numpy arrays
    :param sections: list of indices to split at (not sizes of each split)

    :returns: list of trees with length `len(sections)` with the same shape as x
            where each leaf is the corresponding section of the leaf in x
    """
    result = []
    start = 0
    for end in sections:
        select_tree = multimap(lambda arr: arr[start:end], x)
        start = end
        result.append(select_tree)
    return result


def dtype(tt: TensorType) -> np.dtype:
    """
    :param tt: TensorType to get dtype for

    :returns: numpy.dtype to use for tt
    """
    assert isinstance(tt, TensorType)
    return np.dtype(tt.eltype.dtype_name)


def zeros(vt: ValType, bshape: Tuple) -> Any:
    """
    :param vt: ValType to create zeros for
    :param bshape: batch shape to prepend to the shape of each numpy array created by this function

    :returns: tree of numpy arrays matching vt
    """
    return multimap(
        lambda subdt: np.zeros(bshape + subdt.shape, dtype=dtype(subdt)), vt
    )


def _sample_tensor(
    tt: TensorType, bshape: Tuple, rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    :param tt: TensorType to create sample for
    :param bshape: batch shape to prepend to the shape of each numpy array created by this function
    :param rng: np.random.RandomState to use for sampling

    :returns: numpy array matching tt
    """
    if rng is None:
        rng = np.random
    assert isinstance(tt, TensorType)
    eltype = tt.eltype
    shape = bshape + tt.shape
    if isinstance(eltype, Discrete):
        return rng.randint(eltype.n, size=shape, dtype=dtype(tt))
    elif isinstance(eltype, Real):
        return rng.randn(*shape).astype(dtype(tt))
    else:
        raise ValueError(f"Expected ScalarType, got {type(eltype)}")


def sample(
    vt: ValType, bshape: Tuple, rng: Optional[np.random.RandomState] = None
) -> Any:
    """
    :param vt: ValType to create sample for
    :param bshape: batch shape to prepend to the shape of each numpy array created by this function
    :param rng: np.random.RandomState to use for sampling

    :returns: tree of numpy arrays matching vt
    """
    return multimap(partial(_sample_tensor, bshape=bshape, rng=rng), vt)
