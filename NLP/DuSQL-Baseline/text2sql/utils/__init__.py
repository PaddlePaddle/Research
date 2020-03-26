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

from text2sql.utils.nn_utils import PaddleFluidWrapper as fluider

import numpy as np

def pad_batch_data(insts,
                   max_len=None,
                   insts_data_type="int64",
                   shape=None,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   paddle_version_code=1.5):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if max_len is None:
        max_len = max(len(inst) for inst in insts) if shape is None else shape[1]

    if shape is None:
        shape = [-1, max_len, 1]

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape(shape)]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def fix_random_seed(seed, trainer):
    """固定主要随机数的种子，保证实验可复现

    Args:
        seed (TYPE): NULL
        trainer (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if args.seed is None:
        return False

    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ['FLAGS_cudnn_deterministic'] = 'True'
    return True

