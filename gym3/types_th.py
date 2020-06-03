"""
This is the torch equivalent of the numpy types_np.py

This can be used to perform the same operations but on torch tensors instead of numpy arrays.
"""

from functools import partial
from typing import Any, Sequence, Tuple

import torch as th

from gym3.types import Discrete, Real, TensorType, ValType, multimap


def concat(xs: Sequence[Any], dim: int = 0) -> Any:
    """
    Concatenate the (leaf) tensors from xs

    :param xs: list of trees with the same shape, where the leaf values are torch tensors
    :param dim: dimension to concatenate along
    """
    return multimap(lambda *xs: th.cat(xs, dim=dim), *xs)


def stack(xs: Sequence[Any], dim: int = 0) -> Any:
    """
    Stack the (leaf) tensors from xs

    :param xs: list of trees with the same shape, where the leaf values are torch tensors
    :param dim: dimension to stack along
    """
    return multimap(lambda *xs: th.stack(xs, dim=dim), *xs)


def split(x: Any, sections: Sequence[int]) -> Any:
    """
    Split the (leaf) tensors from the tree x

    Examples:

        split([1,2,3,4], [1,2,3,4]) => [[1], [2], [3], [4]]
        split([1,2,3,4], [1,3,4]) => [[1], [2, 3], [4]]

    :param x: a tree where the leaf values are torch tensors
    :param sections: list of indices to split at (not sizes of each split)

    :returns: list of trees with length `len(sections)` with the same shape as x
            where each leaf is the corresponding section of the leaf in x
    """
    # split each leaf and select the correct component
    result = []
    start = 0
    for end in sections:
        select_tree = multimap(lambda t: t[start:end], x)
        start = end
        result.append(select_tree)
    return result


def dtype(tt: TensorType) -> th.dtype:
    """
    :param tt: TensorType to get dtype for

    :returns: torch.dtype to use for tt
    """
    assert isinstance(tt, TensorType)
    return getattr(th, tt.eltype.dtype_name)


def zeros(vt: ValType, bshape: Tuple) -> Any:
    """
    :param vt: ValType to create zeros for
    :param bshape: batch shape to prepend to the shape of each tensor created by this function

    :returns: tree of torch tensors matching vt
    """
    return multimap(
        lambda subdt: th.zeros(bshape + subdt.shape, dtype=dtype(subdt)), vt
    )


def _sample_tensor(tt: TensorType, bshape: Tuple) -> Any:
    """
    :param tt: TensorType to create sample for
    :param bshape: batch shape to prepend to the shape of each torch tensor created by this function

    :returns: torch tensor matching tt
    """
    assert isinstance(tt, TensorType)
    eltype = tt.eltype
    shape = bshape + tt.shape
    if isinstance(eltype, Discrete):
        return th.randint(0, eltype.n, size=shape, dtype=dtype(tt))
    elif isinstance(eltype, Real):
        return th.randn(*shape, dtype=dtype(tt))
    else:
        raise ValueError(f"Expected ScalarType, got {type(eltype)}")


def sample(vt: ValType, bshape: Tuple) -> Any:
    """
    :param vt: ValType to create sample for
    :param bshape: batch shape to prepend to the shape of each torch tensor created by this function

    :returns: tree of torch tensors matching vt
    """
    return multimap(partial(_sample_tensor, bshape=bshape), vt)
