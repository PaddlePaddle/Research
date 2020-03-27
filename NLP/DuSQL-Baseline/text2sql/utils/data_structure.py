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

"""basic data structure wrapper for tensor in paddle.
like stack, array.
"""

import sys
import os
import traceback
import logging
from collections import namedtuple

import numpy as np
from paddle import fluid
from paddle.fluid import layers

from text2sql.utils import fluider
from text2sql.utils import nn_utils

ArrayData = namedtuple("ArrayData", "data pos")
StackData = namedtuple("StackData", "data pos")

class Array(object):

    """Array function simulator"""

    def __init__(self):
        """init of class """
        super(Array, self).__init__()

    @classmethod
    def push(cls, array_data, updates, in_place=True):
        """append udpates to array_data.data on array_data.pos

        Args:
            array_data (TYPE): NULL
            updates (TYPE): NULL
            in_place (bool): 默认是 True.

        Returns: None

        Raises: NULL
        """
        new_data = nn_utils.batch_scatter(array_data.data, array_data.pos, updates, overwrite=True, in_place=in_place)
        new_pos = fluider.increment(array_data.pos, value=1, in_place=in_place)
        if in_place:
            return array_data
        else:
            return ArrayData(new_data, new_pos)


class Stack(object):

    """Stack function simulator"""

    def __init__(self):
        """init of class """
        super(Stack, self).__init__()

    @classmethod
    def pop(cls, stack_data, mask=True, in_place=True):
        """pop data in stack_data

        Args:
            stack_data (StackData): (data, pos) with shape ([batch_size, stack_len], [batch_size, 1])
            mask (bool): 是否 mask 空栈的返回值。默认为 True
            in_place (bool): 默认为 True

        Returns: (Variable1, Variable2)
            Variable1: pop 得到的值
                       dtype=stack_data.data.dtype
                       shape=[-1]
            Variable2: 对应位置的值是否合法。入参已经为空的栈，此处为 False。
                       dtype=bool
                       shape=[-1]
        Raises: NULL
        """
        data = stack_data.data
        pos = stack_data.pos

        # 只有非空的栈才能pop（才合法）
        valid_pos = layers.logical_not(cls.empty(stack_data))
        new_pos_delta = layers.cast(valid_pos, dtype=pos.dtype)
        new_pos = layers.elementwise_sub(pos, new_pos_delta)

        # shape = [batch_size]
        output = nn_utils.batch_gather(data, new_pos)
        # mask 空栈的返回值
        if mask:
            # shape = [batch_size, 1]
            mask_tag = layers.cast(new_pos_delta, dtype=data.dtype) if data.dtype != pos.dtype else new_pos_delta
            mask_tag = layers.squeeze(mask_tag, [1])
            output = layers.elementwise_mul(output, mask_tag)

        # 出栈后原位置置为0
        updates = layers.zeros_like(output)
        new_data = nn_utils.batch_scatter(data, new_pos, updates, overwrite=True, in_place=in_place)

        if in_place:
            layers.assign(new_pos, pos)
            return output, valid_pos, stack_data
        else:
            return output, valid_pos, StackData(new_data, new_pos)

    @classmethod
    def push(cls, stack_data, updates, in_place=True):
        """push udpates to stack_data

        Args:
            stack_data (TYPE): NULL
            updates (TYPE): NULL
            in_place (bool): 默认是 True.

        Returns: None

        Raises: NULL
        """
        new_data = nn_utils.batch_scatter(stack_data.data, stack_data.pos, updates, overwrite=True, in_place=in_place)
        new_pos = fluider.increment(stack_data.pos, value=1, in_place=in_place)
        if in_place:
            return stack_data
        else:
            return StackData(new_data, new_pos)

    @classmethod
    def empty(cls, stack_data, dtype='bool'):
        """Return True if stack is empty(pos == 0)

        Args:
            stack_data (TYPE): NULL
            dtype (str): result dtype. Default is bool.

        Returns: Variable
                 shape=[-1], dtype=params<dtype>

        Raises: NULL
        """
        zeros = layers.zeros_like(stack_data.pos)
        output = layers.equal(stack_data.pos, zeros)
        if dtype != 'bool':
            output = layers.cast(output, dtype=dtype)
        return output
        

if __name__ == "__main__":
    """run some simple test cases"""
    pass

