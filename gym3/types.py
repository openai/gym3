from typing import Any, Callable, Tuple

from gym3.internal import misc

INTEGER_DTYPE_NAMES = set(
    ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
)
FLOAT_DTYPE_NAMES = set(["float32", "float64"])
DTYPE_NAME_TO_MAX_VALUE = {}
DTYPE_NAME_TO_BIT_WIDTH = {}
for signed in [True, False]:
    for bit_width in (8, 16, 32, 64):
        if signed:
            max_value = 2 ** (bit_width - 1) - 1
        else:
            max_value = 2 ** bit_width
        DTYPE_NAME_TO_MAX_VALUE[("" if signed else "u") + f"int{bit_width}"] = max_value
        DTYPE_NAME_TO_BIT_WIDTH[("" if signed else "u") + f"int{bit_width}"] = bit_width


def pod_equals(x, y):
    """
    equality for plain-old-data types
    """
    return type(x) == type(y) and x.__dict__ == y.__dict__


class ScalarType:
    """
    Type of a scalar, used as the base class for the element type of TensorTypes
    """

    def __init__(self):
        self.dtype_name = None

    __eq__ = pod_equals


class Real(ScalarType):
    """
    A scalar that can represent continuous values
    """

    def __init__(self, dtype_name: str = "float32"):
        self.dtype_name = dtype_name

    def __repr__(self):
        return "<Real>"

    def __str__(self):
        return "R"


class Discrete(ScalarType):
    """
    A scalar that can represent discrete (integer) values

    :param n: number of discrete values that this scalar can assume, which are required to be in the range [0, n)
    :param bit_width: number of bits for the integer dtype that will be used to store values
    :param signed: Set to False to use an unsigned integer type
    """

    def __init__(self, n: int, dtype_name: str = "int64") -> None:
        self.n = int(n)
        self.dtype_name = dtype_name
        assert self.dtype_name in INTEGER_DTYPE_NAMES
        max_value = DTYPE_NAME_TO_MAX_VALUE[self.dtype_name]
        assert (
            n <= max_value + 1
        ), f"{n} cannot be greater than {max_value + 1} for dtype_name={dtype_name}"

    def __repr__(self):
        return f"<Discrete: {self}>"

    def __str__(self):
        return f"D{self.n}"


class ValType:
    """
    Tensor or combination of tensors
    """

    __eq__ = pod_equals


class TensorType(ValType):
    """
    A tensor value type

    :param eltype: instance of ScalarType subclass that represents the types of values in this tensor
    :param shape: shape of the tensor as a tuple of ints
    """

    def __init__(self, eltype: ScalarType, shape: Tuple) -> None:
        assert isinstance(shape, tuple)
        self.eltype = eltype
        self.shape = shape

    @property
    def ndim(self):
        """
        Number of dimensions of the tensor
        """
        return len(self.shape)

    @property
    def size(self):
        """
        Number of elements of the tensor
        """
        return int(misc.intprod(self.shape))

    def __repr__(self):
        return f"<TensorType: {self}>"

    def __str__(self):
        shape_str = ",".join(map(str, self.shape))
        return f"{self.eltype}[{shape_str}]"


def discrete_scalar(n: int) -> TensorType:
    """
    Convenience method for definiting a Discrete TensorType
    """
    return TensorType(shape=(), eltype=Discrete(n))


class DictType(ValType):
    """
    A value type representing a (possibly nested) dictionary of strings to value types
    """

    def __init__(self, **name2type: ValType) -> None:
        self._n2t = name2type

    def __repr__(self):
        return f"<DictType: {self}>"

    def __str__(self):
        elems_str = ", ".join([f"{k}={v}" for (k, v) in self._n2t.items()])
        return f"Dict({elems_str})"

    def __len__(self):
        return len(self._n2t)

    def keys(self):
        return self._n2t.keys()

    def values(self):
        return self._n2t.values()

    def items(self):
        return self._n2t.items()

    def __getitem__(self, key):
        return self._n2t[key]

    def __contains__(self, key):
        return key in self._n2t


def multimap(f: Callable, *xs: Any) -> Any:
    """
    Apply f at each leaf of the list of trees

    A tree is:
        * a (possibly nested) dict
        * a (possibly nested) DictType
        * any other object (a leaf value)

    `{"a": 1}`, `{"a": {"b": 2}}`, and `3` are all valid trees, where the leaf values
    are the integers

    :param f: function to call at each leaf, must take len(xs) arguments
    :param xs: a list of trees, all with the same structure

    :returns: A tree of the same structure, where each leaf contains f's return value.
    """
    first = xs[0]
    if isinstance(first, dict) or isinstance(first, DictType):
        assert all(isinstance(x, dict) or isinstance(x, DictType) for x in xs)
        assert all(sorted(x.keys()) == sorted(first.keys()) for x in xs)
        return {k: multimap(f, *(x[k] for x in xs)) for k in sorted(first.keys())}
    else:
        return f(*xs)
