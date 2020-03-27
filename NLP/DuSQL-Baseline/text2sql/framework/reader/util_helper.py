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
"""
:py:`util_helper`
"""
import paddle
import numpy as np

from text2sql.framework.reader.field import Field
from text2sql.framework.utils.util_helper import truncation_words


def convert_text_to_id(text, field_config):
    """将一个明文样本转换成id
    :param text: 明文文本
    :param field_config : Field类型
    :return:
    """
    if not text:
        raise ValueError("text input is None")
    if not isinstance(field_config, Field):
        raise TypeError("field_config input is must be Field class")

    if field_config.need_convert:
        tokenizer = field_config.tokenizer
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        ids = text.split(" ")

    # 加上截断策略
    if len(ids) > field_config.max_seq_len:
        ids = truncation_words(ids, field_config.max_seq_len, field_config.truncation_type)

    return ids


def padding_batch_data(insts,
                   pad_idx=0,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    :param insts:
    :param pad_idx:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_seq_lens:
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def mask_batch_data(insts, return_seq_lens=False, paddle_version_code=1.6):
    """
    :param insts:
    :param return_seq_lens:
    :param paddle_version_code:
    :return:
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)

    input_mask_data = np.array([[1] * len(inst) + [0] *
                                (max_len - len(inst)) for inst in insts])
    input_mask_data = np.expand_dims(input_mask_data, axis=-1)
    return_list += [input_mask_data.astype("float32")]

    if return_seq_lens:
        if paddle_version_code <= 1.5:
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
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
        if paddle.__version__[:3] <= '1.5':
            seq_lens_type = [-1, 1]
        else:
            seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


def generate_pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False,
                   paddle_version_code=1.6):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # id
    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
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

