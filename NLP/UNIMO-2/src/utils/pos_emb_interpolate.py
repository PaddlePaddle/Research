#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
File: pos_emb_interpolate.py
Author: liwei(liwei85@baidu.com)
Date: 2021-10-15 15:02
Desc:
"""
import paddle


def interpolate_pos_embed(pos_embed_checkpoint, num_patches):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :1, :]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, 1:, :]
        pos_tokens = paddle.reshape(pos_tokens, shape=[-1, orig_size, orig_size, embedding_size])
        pos_tokens = paddle.transpose(pos_tokens, perm=[0, 3, 1, 2])
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = paddle.transpose(pos_tokens, perm=[0, 2, 3, 1])
        pos_tokens = paddle.reshape(pos_tokens, shape=[-1, new_size*new_size, embedding_size])
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
