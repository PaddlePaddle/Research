"""
paddle tensor
"""
from functools import partial

import numpy as np
from collections.abc import Iterable

from . import paddle_delegate_func
from .functional import *
import paddle

"""
paddle tensor
"""
import types
import paddle
from paddle import Tensor

# just for type hint. If there are statements like isinstance(x, FloatTensor), this may cause error
FloatTensor = Tensor

def size(self, dim=None):
    shape = self.shape
    if dim is None:
        return shape
    else:
        return shape[dim]


# def __new__(cls, *args, **kwargs):
#
#     obj = cls.__default_new__(cls, *args, **kwargs)
#
#     setattr(obj, "size", types.MethodType(size, obj))
#
#     return obj
#
# setattr(Tensor, "__default_new__", Tensor.__new__)
# setattr(Tensor, "__new__", __new__)


def bool_(self):
    return self.astype("bool")

def float_(self):
    return self.astype('float32')


def double_(self):
    return self.astype("float64")


def int_(self):
    return self.astype("int32")


def long_(self):
    return self.astype('int64')


def expand(self, *sizes):
    if isinstance(sizes[0], Iterable):
        sizes = sizes[0]
    ##handle -1 case
    if len(sizes) > len(self.shape):
        for _ in range(len(sizes) - len(self.shape)):
            self = self.unsqueeze(dim=0)
    expand_times = [x // y if x >= y else 1 for x, y in zip(sizes, self.shape)]
    x = paddle.fluid.layers.expand(self, expand_times, name=None)
    return x


def masked_fill(self, mask, value):
    if self.ndim == 0:
        assert mask.ndim == 0
        if mask.item():
            return paddle.full([], value, self.dtype) 
        else:
            return self 

    y = paddle.full(self.shape, value, self.dtype)
    mask_shape = [1] * (self.ndim - mask.ndim) + mask.shape
    mask = paddle.reshape(mask, mask_shape)
    mask = paddle.expand_as(mask, self)
    new_values = paddle.where(mask, y, self)
    return new_values
    # mask_float = mask.astype("float32")
    # if self.dtype == paddle.bool:
    #     self_float = self.astype("float32")
    # else:
    #     self_float = self
    # result = self_float * (1 - mask_float) + mask_float * value
    # if self.dtype == paddle.bool:
    #     result = result.astype(paddle.bool)
    # return result

# def masked_fill_(self, mask, value):
#
#     new_values = masked_fill(self, mask, value)
#     paddle.assign(new_values, self)
#
#     return self


def to(self, arg):
    if isinstance(arg, paddle.dtype):
        return self.astype(arg)
    elif isinstance(arg, Tensor):
        return self.astype(arg.dtype)
    else:
        return self

def is_floating_point(self):
    return self.dtype in {paddle.float16, paddle.float32, paddle.float64}


def reshape(self, *size):

    if len(size) == 1 and isinstance(size[0], Iterable):
        size = size[0]

    return paddle.reshape(self, size)


def view(self, *size):
    if len(size) == 1 and isinstance(size[0], Iterable):
        size = size[0]

    return reshape(self, size)

def view_as(self, other):

    return view(self, *other.size())


Tensor.__native__size = Tensor.size

Tensor.device = None
Tensor.float = float_
Tensor.double = double_
Tensor.int = int_
Tensor.long = long_
Tensor.bool = bool_
Tensor.scatter_explicit_index = Tensor.scatter
Tensor.scatter = scatter
Tensor.scatter_explicit_index_ = Tensor.scatter_
Tensor.scatter_ = scatter_
Tensor.scatter_add = scatter_add
Tensor.scatter_add_ = scatter_add_
Tensor.expand = expand
Tensor.masked_fill = masked_fill
#Tensor.masked_fill_ = masked_fill_
Tensor.to = to
Tensor.is_floating_point = is_floating_point
Tensor.reshape = reshape
Tensor.view = view
Tensor.view_as = view_as

Tensor.__invert__ = paddle.logical_not

Tensor.__native__numel = Tensor.numel
def numel(x):
    return x.__native__numel().item()

Tensor.numel = numel

import math

class SizeObject(int):

    def __new__(cls, sizes, *args, **kwargs):
        size = int(math.prod(sizes))
        instance = int.__new__(cls, size, *args, **kwargs)
        instance.sizes = sizes
        return instance

    def __call__(self, index=None):
        if index is None:
            return self.sizes
        else:
            return self.sizes[index]

Tensor.size = property(lambda self: SizeObject(self.shape))


def flatten(self, *args, **kwargs):

    if self.dtype == paddle.bool:
        return flatten(self.int(), *args, **kwargs) > 0
    else:
        return paddle.flatten(self, *args, **kwargs)

Tensor.flatten = flatten


Tensor.__getitem__official__ = Tensor.__getitem__

import builtins

def getitem(self, args):

    if self.dtype == paddle.bool:
        return getitem(self.int(), args) > 0

    if isinstance(args, (list, tuple)):
        ellipsis_num = builtins.sum(x is Ellipsis for x in args)
        if ellipsis_num > 1:
            raise Exception(f"multiple ellipsis found in args: {args}")
        elif ellipsis_num == 1:
            args = list(args)
            ellips_idx = args.index(Ellipsis)
            args_before_ellips = args[:ellips_idx]
            args_after_ellips = args[ellips_idx+1:]
            ommited_dims = [builtins.slice(None, None, None) for _ in range(self.ndim - len(args) + 1)]
            args = tuple(args_before_ellips + ommited_dims + args_after_ellips)

        return self.__getitem__official__(args)

    elif isinstance(args, Tensor):
        if args.dtype == paddle.bool and args.ndim > 1:
            # paddle do not support boolean indexing with ndim > 1
            return self.flatten(start_axis=0, stop_axis=args.ndim-1)[args.flatten().nonzero()]
        if args.ndim == 0:
            assert args.dtype == paddle.bool
            assert self.ndim == 0
            return tensor(self.reshape((1,))[args.reshape((1,))].item(), dtype=self.dtype)

    return self.__getitem__official__(args)

Tensor.__getitem__ = getitem

Tensor.__setitem__official__ = Tensor.__setitem__

def setitem(self, index, value):

    if isinstance(index, Tensor):
        if self.ndim == 0:
            index = index.item()
            assert type(index) == bool
            if index:
                self.fill_(value)
            return

        if index.dtype == paddle.bool and (paddle.any(paddle.isnan(self)) or paddle.any(paddle.isinf(self))):

            result = masked_fill(self, index, value)
            self.set_value(result)
            return

    self.__setitem__official__(index, value)

Tensor.__setitem__ = setitem

def getattribute(self, *args, **kwargs):
    # Perform custom logic here

    obj = object.__getattribute__(self, *args, **kwargs)

    if isinstance(obj, types.MethodType) and not obj.__module__.startswith("paddleext.torchapi."):

        return partial(paddle_delegate_func, obj)
    else:
        return obj


Tensor.__getattribute__ = getattribute

Tensor.sum = sum



def permute(self, *perm):

    if len(perm) == 1 and isinstance(perm[0], Iterable):
        perm = perm[0]

    assert len(perm) == self.ndim
    perm = [self.ndim + x if x < 0 else x for x in perm]  ##not allow negative values

    if self.dtype == paddle.bool:
        return permute(self.int(), * perm) > 0
    else:
        return paddle.transpose(self, perm)

Tensor.permute = permute


def transpose(self, *perm):
    # if len(perm)==2 and len(self.shape)>2:
    if isinstance(perm[0], Iterable):
        assert len(perm) == 1
        perm = perm[0]

    if len(perm) == 2 and len(perm) < self.ndim:

        perm = [self.ndim + x if x < 0 else x for x in perm]
        dim1, dim2 = perm
        perm = list(range(self.rank()))
        perm[dim1] = dim2
        perm[dim2] = dim1

        return self.permute(*perm)
    else:
        return paddle.transpose(self, perm)


Tensor.transpose = transpose

def contiguous(self):
    return self

Tensor.contiguous = contiguous


Tensor.__lt__origin__ = Tensor.__lt__
def __lt__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__lt__origin__(other)
Tensor.__lt__ = __lt__


Tensor.__le__origin__ = Tensor.__le__
def __le__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__le__origin__(other)
Tensor.__le__ = __le__


Tensor.__gt__origin__ = Tensor.__gt__
def __gt__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__gt__origin__(other)
Tensor.__gt__ = __gt__


Tensor.__ge__origin__ = Tensor.__ge__
def __ge__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__ge__origin__(other)
Tensor.__ge__ = __ge__


Tensor.__eq__origin__ = Tensor.__eq__
def __eq__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__eq__origin__(other)
Tensor.__eq__ = __eq__


Tensor.__ne__origin__ = Tensor.__ne__
def __ne__(self, other):
    if self.ndim == 0 and np.isscalar(other):
        other = tensor(other)
    return self.__ne__origin__(other)
Tensor.__ne__ = __ne__


def __or__(self, other):
    return paddle.logical_or(self.bool(), other.bool())
Tensor.__or__ = __or__

def __and__(self, other):
    return paddle.logical_or(self.bool(), other.bool())
Tensor.__and__ = __and__


Tensor.__native__any = Tensor.any
def any(x, dim=None, keepdim=False, name=None):
    if isinstance(x, Tensor) and x.ndim == 0:
        assert dim is None
        return x
    else:
        return x.__native__any(axis=dim, keepdim=keepdim, name=name)

Tensor.any = any

Tensor.__native__all = Tensor.all
def all(x, dim=None, keepdim=False, name=None):

    if isinstance(x, Tensor) and x.ndim == 0:
        assert dim is None
        return x
    else:
        return x.__native__all(axis=dim, keepdim=keepdim, name=name)

Tensor.all = all

Tensor.__native__add__ = Tensor.__add__
#Tensor.__native__iadd__ = Tensor.__iadd__
def add(x, y):

    tensor_out = isinstance(x, Tensor) or isinstance(y, Tensor)

    out_dtype = x.dtype if isinstance(x, Tensor) else y.dtype if isinstance(y, Tensor) else None

    if isinstance(x, Tensor) and x.ndim == 0:
        x = x.item()
    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    if isinstance(x, Tensor):
        return Tensor.__native__add__(x, y)
    elif isinstance(y, Tensor):
        return Tensor.__native__add__(y, x)
    else:
        result = x + y
        if np.isscalar(result) and tensor_out:
            return tensor(result, dtype=out_dtype)
        else:
            return result


# def iadd(x, y):
#     if isinstance(y, Tensor) and y.ndim == 0:
#         y = y.item()
#
#     return Tensor.__native__iadd__(x, y)

Tensor.__add__ = add
Tensor.__radd__ = add
# Tensor.__iadd__ = iadd

Tensor.__native__sub__ = Tensor.__sub__
Tensor.__native__rsub__ = Tensor.__rsub__

def subtract(x, y):
    tensor_out = isinstance(x, Tensor) or isinstance(y, Tensor)

    out_dtype = x.dtype if isinstance(x, Tensor) else y.dtype if isinstance(y, Tensor) else None

    if isinstance(x, Tensor) and x.ndim == 0:
        x = x.item()
    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    if isinstance(x, Tensor):
        return Tensor.__native__sub__(x, y)
    elif isinstance(y, Tensor):
        return Tensor.__native__rsub__(y, x)
    else:
        result = x - y
        if np.isscalar(result) and tensor_out:
            return tensor(result, dtype=out_dtype)
        else:
            return result

def rsub(x, y):

    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    return Tensor.__native__rsub__(x, y)


Tensor.__sub__ = subtract
Tensor.__rsub__ = rsub

Tensor.__native__mul__ = Tensor.__mul__
def multiply(x, y):
    tensor_out = isinstance(x, Tensor) or isinstance(y, Tensor)

    out_dtype = x.dtype if isinstance(x, Tensor) else y.dtype if isinstance(y, Tensor) else None

    if isinstance(x, Tensor) and x.ndim == 0:
        x = x.item()
    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    if isinstance(x, Tensor):
        return Tensor.__native__mul__(x, y)
    elif isinstance(y, Tensor):
        return Tensor.__native__mul__(y, x)
    else:
        result = x * y
        if np.isscalar(result) and tensor_out:
            return tensor(result, dtype=out_dtype)
        else:
            return result

Tensor.__mul__ = multiply
Tensor.__rmul__ = multiply

Tensor.__native__truediv__ = Tensor.__truediv__
Tensor.__native__rdiv__ = Tensor.__rdiv__

def divide(x, y):
    tensor_out = isinstance(x, Tensor) or isinstance(y, Tensor)

    out_dtype = x.dtype if isinstance(x, Tensor) else y.dtype if isinstance(y, Tensor) else None

    if isinstance(x, Tensor) and x.ndim == 0:
        x = x.item()
    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    if isinstance(x, Tensor):
        return Tensor.__native__truediv__(x, y)
    elif isinstance(y, Tensor):
        return Tensor.__native__rdiv__(y, x)
    else:
        result = x / y
        if np.isscalar(result) and tensor_out:
            return tensor(result, dtype=out_dtype)
        else:
            return result

def rdiv(x, y):
    if isinstance(y, Tensor) and y.ndim == 0:
        y = y.item()

    return Tensor.__native__rdiv__(x, y)


Tensor.__truediv__ = divide
Tensor.__rdiv__ = rdiv


def __getstate__(self):
    state = {"dtype": self.dtype, "value": self.numpy()}
    return state


def __setstate__(self, newstate):

    loaded = paddle.to_tensor(newstate["value"], dtype=newstate["dtype"])
    self.set_value(loaded)

Tensor.__getstate__ = __getstate__
Tensor.__setstate__ = __setstate__

## requires_grad property

def getter(x):
    return not x.stop_gradient

def setter(x, value):
    x.stop_gradient = not value

Tensor.requires_grad = property(getter, setter)

Tensor.topk = topk