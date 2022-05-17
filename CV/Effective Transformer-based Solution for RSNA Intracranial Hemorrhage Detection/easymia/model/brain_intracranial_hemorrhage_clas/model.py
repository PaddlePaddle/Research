# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################
"""
Swin transformer-> brain  braun intracranial hemorrhage clas
"""

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from easymia.core.abstract_model import AbstractModel
from easymia.libs import manager

from .swin_transformers_update import SwinTransformerV2

@manager.MODELS.add_component
class BrainIntranialHemorrhageClsModel(AbstractModel):
    """
    Model
    """
    def __init__(self, mode, image_size=224, in_ch=1, num_classes=6, sequential=False):
        super(BrainIntranialHemorrhageClsModel, self).__init__(mode)
        self.backbone = SwinTransformerV2(
            image_size=image_size,
            patch_size=4,
            in_channels=in_ch,
            num_classes=num_classes,
            droppath=0.5,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            )
        
        self.sequential = sequential
        if sequential:
            d_model = self.backbone.num_features
            self.pos_encoder = PositionalEncoding(d_model, max_len=60, dropout=0.2) # BrainIHD数据集单个CT最多60个切片
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=200, dropout=0.2,
                                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1)),
                                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)))
            self.sequence_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
            self.sequence_decoder = nn.Linear(d_model, num_classes)

    def __clas__(self, x):
        """
        分类模型方法
        """
        # x: paddle tensor, shape [N, C, H, W], C=1
        if not self.sequential:
            return self.backbone(x)
        else:
            features = self.backbone.forward_features(x) # N, Features
            stage1_out = self.backbone.fc(features)

            src = paddle.unsqueeze(features, axis=0) # 1, Slices, Features
            src = self.pos_encoder(src)

            stage2_out = self.sequence_encoder(src)
            stage2_out = self.sequence_decoder(stage2_out)
            stage2_out = paddle.squeeze(stage2_out, 0) # 1, S, C -> S,C

            # stage2_out_sum = stage2_out.max(0, keepdim=True)

            if self.training:
                return stage2_out, stage1_out
            else:
                return stage2_out


class PositionalEncoding(nn.Layer):
    """
    位置编码，从pytorch转写而来
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros([max_len, 1, d_model])
        pe[:, 0, 0::2] = np.sin(position * div_term)
        pe[:, 0, 1::2] = np.cos(position * div_term)
        pe = pe.transpose(1, 0, 2)
        self.register_buffer('pe', paddle.to_tensor(pe))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)