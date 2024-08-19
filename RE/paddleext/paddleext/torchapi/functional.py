import builtins
from collections import namedtuple

import paddle
from paddle import Tensor
import numpy as np

from paddle import is_tensor

from paddle import less_than, less_equal, greater_than, greater_equal, equal

from paddle.nn.functional import *

from paddle import arange, ones_like, zeros_like, ones

from paddle import logical_and, logical_not, logical_or, logical_xor

from paddle import all, any

from paddle import argmax, argmin

from paddle import stack

from paddle import einsum

from paddle import inverse

from paddle.linalg import * 


def max_along_dim(input, dim=None, keepdim=False, *, out=None):

    if dim is None:
        result = paddle.max(input)
        return paddle.ones([], dtype=result.dtype) * result.item()

    max_val = paddle.max(input, axis=dim, keepdim=keepdim)
    max_index = paddle.argmax(input, axis=dim)

    if out is not None:
        out[0] = max_val
        out[1] = max_index

    return (max_val, max_index)

def max(input, *args, **kwargs):

    if len(args) == 0:
        return max_along_dim(input, **kwargs)

    if isinstance(args[0], (int, list, tuple)):
        return max_along_dim(input, *args, **kwargs)
    elif isinstance(args[0], Tensor):
        return paddle.maximum(input, args[0], *args[1:], **kwargs)
    else:
        raise Exception(f"unknown parameter combination")


def min_along_dim(input, dim=None, keepdim=False, *, out=None):

    if dim is None:
        result = paddle.min(input)
        return paddle.ones([], dtype=result.dtype) * result.item()

    min_val = paddle.min(input, axis=dim, keepdim=keepdim)
    min_index = paddle.argmin(input, axis=dim)

    if out is not None:
        out[0] = min_val
        out[1] = min_index

    return (min_val, min_index)


def min(input, *args, **kwargs):

    if len(args) == 0:
        return min_along_dim(input, **kwargs)

    if isinstance(args[0], (int, list, tuple)):
        return min_along_dim(input, *args, **kwargs)
    elif isinstance(args[0], Tensor):
        return paddle.minimum(input, args[0], *args[1:], **kwargs)
    else:
        raise Exception(f"unknown parameter combination")

 
def lt(a, b):
    if np.isscalar(a) or np.isscalar(b):
        return a < b
    else:
        return less_than(a, b)

 
def le(a, b):
    if np.isscalar(a) or np.isscalar(b):
        return a <= b
    else:
        return less_equal(a, b)

 
def gt(a, b):
    if np.isscalar(a) or np.isscalar(b):
        return a > b
    else:
        return greater_than(a, b)

 
def ge(a, b):
    if np.isscalar(a) or np.isscalar(b):
        return a >= b
    else:
        return greater_equal(a, b)

 
def eq(a, b):
    if np.isscalar(a) or np.isscalar(b):
        return a == b
    else:
        return equal(a, b)


def standardize_dtype(type):

    if type == int:
        return paddle.int64
    elif type == float:
        return paddle.float32

    return type

def empty(*size, dtype=None, device=None):

    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]

    dtype = standardize_dtype(dtype)
    x = paddle.empty(size, dtype=dtype)

    return x

def zeros(*size, dtype=None, device=None):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]

    dtype = standardize_dtype(dtype)
    x = paddle.zeros(size, dtype=dtype)
    # if device is not None:
    #     x = x.to(device)
    return x

 
def ones(*size, dtype=None, device=None):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dtype = standardize_dtype(dtype)
    x = paddle.ones(size, dtype=dtype)
    # if device is not None:
    #     x = x.to(device)
    return x

 
def rand(*size, dtype=None, device=None):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dtype = standardize_dtype(dtype)
    x = paddle.rand(size, dtype=dtype)
    # if device is not None:
    #     x = x.to(device)
    return x

 
def randint(low=None, high=None, size=None, dtype=None, name=None, device=None):

    arg1 = low
    arg2 = high
    arg3 = size

    dtype = standardize_dtype(dtype)

    if dtype == paddle.int32 or dtype == paddle.int64:
        int_dtype = dtype
        target_dtype = None
    else:
        int_dtype = None
        target_dtype = None

    if arg3 is not None:
        assert isinstance(arg3, (list, tuple))
        if low is None and high is not None:
            arg1 = high
            arg2 = None
        result = paddle.randint(low=arg1, high=arg2, shape=arg3, dtype=int_dtype, name=name)
        return result.astype(target_dtype) if target_dtype else result
    else:
        assert isinstance(arg2, (list, tuple))
        result = paddle.randint(low=arg1, high=None, shape=arg2, dtype=int_dtype, name=name)
        return result.astype(target_dtype) if target_dtype else result


def randn(*size, out=None, dtype=None, device=None):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]

    dtype = standardize_dtype(dtype)
    x = paddle.randn(size, dtype=dtype)

    if out is not None:
        paddle.assign(x, out)
        return out

    return x


def manual_seed_all(seed):
    paddle.seed(seed)


def manual_seed(seed):
    paddle.seed(seed)


def scalar_dtype(x):
    from . import core
    return getattr(core, type(x).__name__)

 
def tensor(x, dtype=None, device=None):
    if np.isscalar(x):
        if dtype is None:
            dtype = scalar_dtype(x)
        result = paddle.ones([], dtype=dtype)
        if np.isnan(x):
            result = (result * (-1)).sqrt()
        else:
            result.fill_(x)
        return result

    return paddle.to_tensor(x, dtype=dtype)

def from_numpy(x):
    return paddle.to_tensor(x)


cat = paddle.concat


# different meaning of scatter
# in tensorflow/ paddle, scatter is :
# for idx, l in enumerate(index):
#   output[l] = update[idx]
# in torch, scatter is:
# for i, j, k:
#   output[i, j, index[i,j,k]] = update[i, j, k]

 
def sum(x, dim=None, dtype=None, keepdim=False, name=None):

    if x.ndim == 0:
        return x

    result = paddle.sum(x, axis=dim, dtype=dtype, keepdim=keepdim, name=name)

    dim_len = 1 if np.isscalar(dim) else x.ndim if dim is None else len(dim)

    if not keepdim and x.ndim == dim_len:
        return tensor(result.item(), dtype=result.dtype)
    else:
        return result

 
def nonzero(input, *, out=None, as_tuple=False):

    result = paddle.nonzero(input, as_tuple=as_tuple)
    if not as_tuple:
        if out is not None:
            paddle.assign(result, out)
            return out
        else:
            return result
    else:
        assert out is None
        return tuple([x.squeeze(-1) for x in result])

 
def where(condition, x=None, y=None, name=None):

    if x is not None and y is not None:
        assert is_tensor(x) or is_tensor(y)

        if np.isscalar(x):
            x = paddle.ones_like(condition, dtype=scalar_dtype(x)) * x
        if x.ndim == 0:
            x = paddle.ones_like(condition, dtype=x.dtype) * x.item()

        if np.isscalar(y):
            y = paddle.ones_like(condition, dtype=scalar_dtype(y)) * y
        if x.ndim == 0:
            y = paddle.ones_like(condition, dtype=y.dtype) * y.item()

        return paddle.where(condition, x, y, name=name)
        
    elif x is None and y is None:
        result = nonzero(condition, as_tuple=True)

        return result
    else:
        raise Exception("x and y must be None or not None at the sametime")

 
def is_nonzero(input):

    assert paddle.numel(input) == 1

    return input.item() != 0.0

 
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):

    if np.isscalar(input):
        input = tensor(input)
    if np.isscalar(other):
        other = tensor(other)

    return paddle.allclose(input.float(), other.float(), rtol=rtol, atol=atol, equal_nan=equal_nan, name=name)

 
def scatter(input: Tensor, dim, index, value) -> Tensor:
    if input.ndim == 1:
        output = paddle.scatter(input, index, value, overwrite=True)
    else:

#        index, selected = paddle.unique(index, axis=dim, return_index=True)
#        if isinstance(value, Tensor):
#            value = paddle.index_select(value, selected, axis=dim)

        grids = [paddle.arange(index.shape[x]) for x in range(index.ndim)]
        inner_indexes = list(paddle.meshgrid(*grids))
        inner_indexes[dim] = index
        inner_indexes = [x.flatten() for x in inner_indexes]
        inner_indexes = paddle.stack(inner_indexes, axis=1)

        value_shape = list(inner_indexes.shape[:-1]) + list(input.shape[inner_indexes.shape[-1]:])

        if paddle.is_tensor(value):
            value = paddle.reshape(value, value_shape)
        elif isinstance(value, (builtins.bool, builtins.int, builtins.float, np.integer, np.float32, np.float64)):
            value = paddle.full(shape=value_shape, fill_value=value)
        else:
            raise Exception(f"unknown value type: {type(value)}")

        to_overwrite = paddle.scatter_nd(inner_indexes, value, shape=input.shape)
        condition = paddle.scatter_nd(inner_indexes, paddle.ones_like(value), shape=input.shape)
        output = paddle.where(condition > 0, to_overwrite.float(), input.float()).cast(input.dtype)

    return output

def gather(x,dim,index):
    index_shape=index.shape
    index_flatten=index.flatten()
    if dim<0:
        dim=len(x.shape)+dim
    nd_index=[]
    for k in range(len(x.shape)):
        if k==dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape=[1]*len(x.shape)
            reshape_shape[k]=x.shape[k]
            dim_index=paddle.expand( paddle.reshape(paddle.arange(x.shape[k],dtype=index.dtype), reshape_shape), index_shape).flatten()
            nd_index.append(dim_index)

    ind2 = paddle.transpose(paddle.stack(nd_index),[1, 0])
    # ind2 = paddle.stack(nd_index).transpose([1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

 
def scatter_(input: Tensor, dim, index, value):

    output = scatter(input, dim, index, value)
    # return output
    paddle.assign(output, input)

    return input


 
def scatter_add(input: Tensor, dim, index, update) -> Tensor:
    # donot use scatter with overwrite=False even for 1-d case;
    # It does not produce correct result for duplicated indexes
    # if input.ndim == 1:
    #     output = paddle.scatter(input, index, update, overwrite=False)
    # else:
    if index.ndim > 1:
        grids = [paddle.arange(index.shape[x]) for x in range(index.ndim)]
        inner_indexes = list(paddle.meshgrid(*grids))
        inner_indexes[dim] = index
    else:
        inner_indexes = [index]
    inner_indexes = [x.flatten() for x in inner_indexes]
    inner_indexes = paddle.stack(inner_indexes, axis=1)

    update_shape = list(inner_indexes.shape[:-1]) + list(input.shape[inner_indexes.shape[-1]:])
    update = paddle.reshape(update, update_shape)
    output = paddle.scatter_nd_add(input, inner_indexes, update)

    return output

 
def scatter_add_(input: Tensor, dim, index, update) -> Tensor:
    output = scatter_add(input, dim, index, update)
    paddle.assign(output, input)
    # return output
    return input


def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):

    result = paddle.linalg.norm(input, p, axis=dim, keepdim=keepdim)
    if dtype is not None:
        result = result.cast(dtype)

    if out is not None:
        out.assign(result)

    return result

def isinf(x, name=None):
    if x.dtype == paddle.bool:
        return paddle.zeros_like(x, dtype=paddle.bool)
    else:
        return paddle.isinf(x, name=name)

def isnan(x, name=None):
    if x.dtype == paddle.bool:
        return paddle.zeros_like(x, dtype=paddle.bool)
    else:
        return paddle.isnan(x, name=name)

def broadcast_to(x, shape, name=None):

    if len(shape) == 1 and shape[0] == 0:
        assert x.numel() == 1
        return tensor(x.item())
    else:
        return paddle.broadcast_to(x, shape, name)


def as_tensor(data, dtype=None, device=None):

    return paddle.to_tensor(data, dtype=dtype)


TopKResult = namedtuple("TopKResult", ["values", "indices"])
def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):

    result, indice = paddle.topk(input, k, axis=dim, largest=largest, sorted=sorted)

    if out is not None:
        out[0].set_value(result)
        out[1].set_value(indice)

    return TopKResult(values=result, indices=indice)


def split(tensor, split_size_or_sections, dim=0):
    """
    paddle interface is different from pytorch

    Args:
        tensor:
        split_size_or_sections:
        dim:

    Returns:

    """
    if isinstance(split_size_or_sections, int):
        sizes = [split_size_or_sections] * (tensor.shape[dim] // split_size_or_sections)
        if tensor.shape[dim] % split_size_or_sections != 0:
            sizes.append(tensor.shape[dim] % split_size_or_sections)
        split_size_or_sections = sizes

    return paddle.split(tensor, split_size_or_sections, axis=dim)