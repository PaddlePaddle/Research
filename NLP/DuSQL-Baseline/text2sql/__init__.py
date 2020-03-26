# -*- coding: utf-8 -*-
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
"""text2sql"""

from text2sql.global_config import GlobalConfig
g = GlobalConfig()

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import layers
from paddle.fluid.layers import nn

def scatter_nd_add_fix_bug(ref, index, updates, name=None):
    """fix bug of paddle.fluid.layers.scatter_nd_add

    Args:
        ref (TYPE): NULL
        index (TYPE): NULL
        updates (TYPE): NULL
        name (TYPE): Default is None

    Returns: TODO

    Raises: NULL
    """
    if ref.dtype != updates.dtype:
        raise ValueError("ref and updates must have same data type.")

    helper = LayerHelper('scatter_nd_add', **locals())
    dtype = helper.input_dtype(input_param_name='ref')
    if name is None:
        output = helper.create_variable_for_type_inference(dtype)
    else:
        output = helper.create_variable(
            name=name, dtype=dtype, persistable=False)
    helper.append_op(
        type="scatter_nd_add",
        inputs={"X": ref,
                "Index": index,
                "Updates": updates},
        outputs={"Out": output})
    return output

layers.scatter_nd_add = scatter_nd_add_fix_bug
nn.scatter_nd_add = scatter_nd_add_fix_bug

