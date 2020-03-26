#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""neural network utils based on paddle fluid
"""

import sys
import os
import traceback
import logging
import functools

import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.layers import control_flow

uniform = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x) if x > 0 else None
zero = fluid.initializer.Constant(0.0)

class PaddleVarType(object):

    """wrapper of paddle var type"""
    bool = fluid.core.VarDesc.VarType.BOOL
    uint8 = fluid.core.VarDesc.VarType.UINT8
    int8 = fluid.core.VarDesc.VarType.INT8
    int16 = fluid.core.VarDesc.VarType.INT16
    int32 = fluid.core.VarDesc.VarType.INT32
    int64 = fluid.core.VarDesc.VarType.INT64
    fp16 = fluid.core.VarDesc.VarType.FP16
    fp32 = fluid.core.VarDesc.VarType.FP32
    fp64 = fluid.core.VarDesc.VarType.FP64
    ints = set([fluid.core.VarDesc.VarType.INT8,
                fluid.core.VarDesc.VarType.INT16,
                fluid.core.VarDesc.VarType.INT32])
    floats = set([fluid.core.VarDesc.VarType.FP16,
                  fluid.core.VarDesc.VarType.FP32,
                  fluid.core.VarDesc.VarType.FP64])


class SupportBool(object):

    """support bool dtype layers by cast it to int32"""

    def __init__(self, layer_fn, *args, **kwargs):
        """init of class

        Args:
            layer_fn (TYPE): NULL
            *args (TYPE): NULL
            **kwargs (TYPE): NULL

        """
        super(SupportBool, self).__init__()

        self._layer_fn = layer_fn
        self._args = args
        self._kwargs = kwargs
        self._is_bool = False
        
    def __enter__(self):
        """enter: cast bool to int32
        Returns: TODO

        Raises: NULL
        """
        pass


class PaddleFluidWrapper(object):
    """wrapper of some paddle fluid layers and other things"""

    dtype = PaddleVarType()

    @classmethod
    def logical_and(cls, x, y, *args, out=None, name=None):
        """wrapper of paddle.fluid.layers.logical_and

        Args:
            x (Variable): NULL
            y (Variable): NULL
            *args (TYPE): NULL
            out (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL

        """
        tmp = layers.logical_and(x, y, out=out, name=name)
        for var in args:
            tmp = layers.logical_and(tmp, var, out=out, name=name)
        return tmp

    @classmethod
    def logical_or(cls, x, y, *args, out=None, name=None):
        """wrapper of paddle.fluid.layers.logical_or

        Args:
            x (Variable): NULL
            y (Variable): NULL
            *args (TYPE): NULL
            out (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL

        """
        tmp = layers.logical_or(x, y, out=out, name=name)
        for var in args:
            tmp = layers.logical_or(tmp, var, out=out, name=name)
        return tmp

    @classmethod
    def elementwise_op_wrapper(cls, op, x, y, *args, force=False, axis=-1, act=None, name=None):
        """wrapper of elementwise op

        Args:
            op (TYPE): NULL
            x (TYPE): NULL
            y (TYPE): NULL
            *args (TYPE): NULL
            force (TYPE): Default is False
            axis (TYPE): Default is -1
            act (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        x_dtype = x.dtype
        if x_dtype == PaddleVarType.bool:
            x = layers.cast(x, dtype='int32')
        tmp = x
        extras = [y] + list(args)
        for var in extras:
            if var.dtype != tmp.dtype and force:
                var = layers.cast(var, dtype=x.dtype)
            elif var.dtype == PaddleVarType.bool and x_dtype == PaddleVarType.bool:
                var = layers.cast(var, dtype=x.dtype)
            tmp = op(x=tmp, y=var, axis=axis, act=act, name=name)
        if x_dtype == PaddleVarType.bool:
            tmp = layers.cast(tmp, dtype=x_dtype)
        return tmp

    @classmethod
    def elementwise_add(cls, x, y, *args, force=False, axis=-1, act=None, name=None):
        """wrapper of paddle.fluid.layers.elementwise_add

        Args:
            x (TYPE): NULL
            y (TYPE): NULL
            *args (TYPE): NULL
            force (TYPE): Default is False
            axis (TYPE): Default is -1
            act (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL

        """
        return cls.elementwise_op_wrapper(layers.elementwise_add, x, y, *args,
                                          force=force, axis=axis, act=act, name=name)

    @classmethod
    def elementwise_sub(cls, x, y, *args, force=False, axis=-1, act=None, name=None):
        """wrapper of paddle.fluid.layers.elementwise_sub

        Args:
            x (TYPE): NULL
            y (TYPE): NULL
            *args (TYPE): NULL
            force (TYPE): Default is False
            axis (TYPE): Default is -1
            act (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        return cls.elementwise_op_wrapper(layers.elementwise_sub, x, y, *args,
                                          force=force, axis=axis, act=act, name=name)

    @classmethod
    def elementwise_mul(cls, x, y, *args, force=False, axis=-1, act=None, name=None):
        """wrapper of paddle.fluid.layers.elementwise_mul

        Args:
            x (TYPE): NULL
            y (TYPE): NULL
            *args (TYPE): NULL
            force (TYPE): Default is False
            axis (TYPE): Default is -1
            act (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        return cls.elementwise_op_wrapper(layers.elementwise_mul, x, y, *args,
                                          force=force, axis=axis, act=act, name=name)

    @classmethod
    def elementwise_div(cls, x, y, *args, force=False, axis=-1, act=None, name=None):
        """wrapper of paddle.fluid.layers.elementwise_div

        Args:
            x (TYPE): NULL
            y (TYPE): NULL
            *args (TYPE): NULL
            force (TYPE): Default is False
            axis (TYPE): Default is -1
            act (TYPE): Default is None
            name (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        return cls.elementwise_op_wrapper(layers.elementwise_div, x, y, *args,
                                          force=force, axis=axis, act=act, name=name)

    @classmethod
    def reshape(cls, x, shape, **kwargs):
        """wrapper of layers.reshape, to support reshape bool layers

        Args:
            x (TYPE): NULL
            shape (TYPE): NULL
            **kwargs (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        input_dtype = x.dtype
        if input_dtype == PaddleVarType.bool:
            x = layers.cast(x, dtype='int32')
        output = layers.reshape(x, shape=shape, **kwargs)
        if input_dtype == PaddleVarType.bool:
            output = layers.cast(output, dtype='bool')
        return output

    @classmethod
    def squeeze(cls, input, axes, name=None):
        """wrapper of layers.gather, to support squeeze bool layers

        Args:
            input (TYPE): NULL
            axes (TYPE): NULL
            name (str): Default is None

        Returns: Variable

        Raises: NULL
        """
        input_dtype = input.dtype
        if input_dtype == PaddleVarType.bool:
            input = layers.cast(input, dtype='int32')
        output = layers.squeeze(input, axes, name)
        if input_dtype == PaddleVarType.bool:
            output = layers.cast(output, dtype='bool')
        return output

    @classmethod
    def gather(cls, input, index, overwrite=True):
        """wrapper of layers.gather, to support gather bool layers

        Args:
            input (TYPE): NULL
            index (TYPE): NULL
            overwrite (TYPE): Default is True

        Returns: Variable

        Raises: NULL
        """
        input_dtype = input.dtype
        if input_dtype == PaddleVarType.bool:
            input = layers.cast(input, dtype='int32')
        output = layers.gather(input, index, overwrite)
        if input_dtype == PaddleVarType.bool:
            output = layers.cast(output, dtype='bool')
        return output

    @classmethod
    def increment(cls, x, value, in_place=False):
        """increment each element in x by value

        Args:
            x (Variable): NULL
            value (int/float): NULL
            in_place (TYPE): Default is False

        Returns: TODO

        Raises: NULL
        """
        if len(x.shape) == 1 and x.shape[0] == 1:
            return layers.increment(x, value, in_place)

        value_tensor = layers.fill_constant(shape=[1], dtype=x.dtype, value=value)
        y = layers.elementwise_add(x, value_tensor)
        if in_place:
            y = layers.assign(y, x)
            return x
        else:
            return y


def batch_gather(var, indices):
    """Gather slices from var in each batch, according to corrensponding
    index in indices. Currently, it only support 2d Tensor.

    Args:
        var (Variable): with shape [batch_size, ...]
        indices (Variable): with shape [batch_size, 1] or [batch_size]

    Returns: Variable with shape [batch_size]

    Raises: NULL

    Examples:
        var
            [[1, 2, 3],
             [4, 5, 6]]
        indices
            [[2], [1]]

        return
            [[3], [5]]

    """
    if len(indices.shape) >= 2 and indices.shape[-1] != 1:
        raise ValueError('shape of indices error. it should be a 1-D layers, or a 2-D layers which '
                         'the 2nd dimension is 1. but got shape = %s' % (str(indices.shape), ))

    if len(indices.shape) == 1:
        indices = layers.reshape(indices, shape=[-1, 1])

    reshape_input = len(var.shape) == 1
    if reshape_input:
        var = PaddleFluidWrapper.reshape(var, shape=[-1, 1])

    batch_size = layers.cast(layers.shape(indices)[0], dtype=indices.dtype)
    zero = layers.fill_constant(shape=[1], dtype=indices.dtype, value=0)
    one = layers.fill_constant(shape=[1], dtype=indices.dtype, value=1)
    batch_indices = layers.unsqueeze(layers.range(zero, batch_size, one, dtype=indices.dtype), [1])

    coord = layers.concat([batch_indices, indices], axis=1)
    coord.stop_gradient = True
    output = layers.gather_nd(var, coord)
    if reshape_input:
        output = PaddleFluidWrapper.reshape(output, shape=[-1])
    return output


def batch_gather_2d(var, indices):
    """Gather slices from var in each batch, according to corrensponding
    index in indices. Currently, it only support 2d Tensor.

    Args:
        var (Variable): with shape [batch_size, ...]
        indices (Variable): with shape [batch_size, max_len]

    Returns: Variable with shape [batch_size]

    Raises: NULL

    Examples:
        var
            [[1, 2, 3],
             [4, 5, 6]]
        indices
            [[2, 0], [1, 2]]

        return
            [[3, 1], [5, 6]]

    """
    if len(indices.shape) != 2:
        raise ValueError('shape of indices error. it should be a 2-D layers. '
                         'but got shape = %s' % (str(indices.shape), ))

    batch_size = layers.shape(indices)[0]

    zero = layers.fill_constant(shape=[1], dtype=indices.dtype, value=0)
    one = layers.fill_constant(shape=[1], dtype=indices.dtype, value=1)
    end = layers.cast(batch_size, dtype=indices.dtype)
    batch_indices_1d = layers.unsqueeze(layers.range(zero, end, one, dtype=indices.dtype), [1])

    seq_len = indices.shape[1]
    batch_indices = layers.expand(batch_indices_1d, [1, seq_len])

    coord_2d = layers.concat([layers.unsqueeze(batch_indices, [2]), layers.unsqueeze(indices, [2])], axis=2)
    coord_2d.stop_gradient = True
    coord_1d = layers.reshape(coord_2d, shape=[-1, 2])
    output_1d = layers.gather_nd(var, coord_1d)
    output_2d = layers.reshape(output_1d, [batch_size, seq_len, var.shape[-1]])
    return output_2d


def batch_scatter(ref, indices, updates, in_place=False, overwrite=False):
    """Scatter updates to ref, according to corrensponding index in indices
    in each batch. Currently, it only support 2d Tensor.

    Args:
        ref (Variable): with shape [batch_size, ...]
        indices (Variable): with shape [batch_size, 1]
        updates (Variable): with shape [batch_size]
        in_place (bool): if True, scatter result will be assign to ref. otherwise,
                         a new Tensor will be returned. Default is False.
        overwrite (bool): if True, scatter will over write corrensponding elements.
                          Default is False.

    Returns: TODO

    Raises: NULL

    Examples:
        ref
            [[1, 1, 1],
             [1, 1, 1]]
        indices
            [[2], [1]]
        updates
            [2, 3]

        return
            [[1, 1, 2],
             [1, 3, 1]]

    """
    ref_dtype = ref.dtype
    if ref_dtype not in PaddleVarType.floats:
        ref_in = layers.cast(ref, dtype='float32')
    else:
        ref_in = ref

    if updates.dtype != ref_in.dtype:
        updates = layers.cast(updates, dtype=ref_in.dtype)

    batch_size = layers.cast(layers.shape(ref_in)[0], dtype=indices.dtype)
    zero = layers.fill_constant(shape=[1], dtype=indices.dtype, value=0)
    one = layers.fill_constant(shape=[1], dtype=indices.dtype, value=1)
    batch_indices = layers.unsqueeze(layers.range(zero, batch_size, one, dtype=indices.dtype), [1])
    coord = layers.concat([batch_indices, indices], axis=1)
    if overwrite:
        mask = layers.gather_nd(ref_in, coord)
        mask = layers.elementwise_sub(layers.zeros_like(mask), mask)
        ref_in = layers.scatter_nd_add(ref_in, coord, mask)

    output = layers.scatter_nd_add(ref_in, coord, updates)
    if ref_dtype not in PaddleVarType.floats:
        output = layers.cast(output, dtype=ref_dtype)
    if in_place:
        layers.assign(output, ref)
        return ref
    else:
        return output


def slices_assign(ref, ref_indices, update, update_indices, in_place=False, overwrite=False):
    """gather slices from update according to update_indices, and scatter them
    to ref according to ref_indices. Note that both ref_indices and update_indices
    do not including index of batch dimension.

    Args:
        ref (Variable): NULL
        ref_indices (Variable): NULL
        update (Variable): NULL
        update_indices (Variable): NULL
        in_place (bool): if True, scatter result will be assign to ref. otherwise,
                         a new Tensor will be returned. Default is False
        overwrite (bool): if True, scatter will over write corrensponding elements.
                          Default is False.

    Returns: TODO

    Raises: NULL

    """
    updates = batch_gather(update, update_indices)
    output = batch_scatter(ref, ref_indices, updates, in_place, overwrite)
    return output


def merge_first_ndim(x, n=2):
    """merge first n dimension of input x

    Args:
        x (TYPE): NULL
        n (TYPE): Default is 2

    Returns: TODO

    Raises: NULL
    """
    return PaddleFluidWrapper.reshape(x, shape=(-1, ) + x.shape[n:])


def split_first_dim(x, size=1):
    """split first dim to two with shape [-1, size, ...]

    Args:
        x (TYPE): NULL
        size (TYPE): Default is 1

    Returns: TODO

    Raises: NULL
    """
    return PaddleFluidWrapper.reshape(x, shape=(-1, size) + x.shape[1:])


def input_true(x, condition, reverse=False):
    """input instances in x, while corrensponding condition is true

    Args:
        x (Variable): shape = [batch_size, ...]
        condition (Variable): shape = [batch_size, 1]
        reverse (Variable): Default is False

    Returns: TODO

    Raises: NULL
    """
    x_dtype = x.dtype
    if x_dtype == PaddleVarType.bool:
        x = layers.cast(x, dtype='int32')

    if condition.dtype != x.dtype:
        condition = layers.cast(condition, dtype=x.dtype)

    if reverse:
        condition = 1.0 - condition

    output = layers.elementwise_mul(x, condition, axis=0)

    if x_dtype == PaddleVarType.bool:
        output = layers.cast(output, dtype=x_dtype)

    return output


def ifelse(condition, vars_for_true, vars_for_false):
    """output true_var while corrensponding condition is True, else output false_var

    Args:
        condition (TYPE): NULL
        vars_for_true (TYPE): NULL
        vars_for_false (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if len(vars_for_true) != len(vars_for_false):
        raise ValueError('input vars for true and for false should have the same length. but got '
                         'vars_for_true(%d) != vars_for_false(%d)' % (len(vars_for_true), len(vars_for_false)))

    fn_true = functools.partial(input_true, condition=condition)
    true_vars = layers.utils.map_structure(fn_true, vars_for_true)

    fn_false = functools.partial(input_true, condition=condition, reverse=True)
    false_vars = layers.utils.map_structure(fn_false, vars_for_false)

    output_vars = layers.utils.map_structure(PaddleFluidWrapper.elementwise_add, true_vars, false_vars)
    return output_vars


def param_attr(name, scale, need_bias=False, method='default'):
    """create param attr

    Args:
        name (str): NULL
        scale (float): param init scale
        need_bias (bool): Default is False
        method (stt): Default is 'default'

    Returns: TODO

    Raises: NULL
    """
    if method != 'default':
        raise ValueError('only <default> method is supported')

    p_init = None
    if scale > 0:
        p_init = uniform(scale)
    param_attr = fluid.ParamAttr(name=name + '_w', initializer=p_init)
    bias_attr = False
    if need_bias:
        bias_attr = fluid.ParamAttr(name=name + '_b', initializer=zero)

    return {'param_attr': param_attr, 'bias_attr': bias_attr}


if __name__ == "__main__":
    """run some simple test cases"""
    from text2sql.utils.debug import executor
    exe = executor.Executor()

    ref = fluid.layers.data(name='ref', shape=[-1, 3], dtype='float32')
    ref_idx = fluid.layers.data(name='ref_idx', shape=[-1, 1], dtype='int64')
    update = fluid.layers.data(name='update', shape=[-1, 2], dtype='float32')
    update_idx = fluid.layers.data(name='update_idx', shape=[-1, 1], dtype='int64')

    out = slices_assign(ref, ref_idx, update, update_idx, overwrite=True)

    def _data():
        return {
                "ref": np.array([[1, 1, 1], [1, 1, 1]]).astype(np.float32),
                "ref_idx": np.array([[2], [1]]).astype(np.int64),
                "update": np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32),
                "update_idx": np.array([[1], [2]]).astype(np.int64),
            }

    result = exe.run(feed=_data(), fetch_list=[out])
    for var in result:
        print(var)

