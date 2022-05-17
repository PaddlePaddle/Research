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
二元交叉熵
"""
import numpy as np
import paddle
import paddle.nn.functional as F

from easymia.core.abstract_loss import AbstractLoss
from easymia.libs import manager

@manager.LOSSES.add_component
class BinaryCrossEntropyLoss(AbstractLoss):
    
    """
    Implements the binary cross entropy loss function.
    Args:
    weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
    given to each class. Its length must be equal to the number of classes.
    Default ``None``.
    ignore_index (int64, optional): Specifies a target value that is ignored
    and does not contribute to the input gradient. Default ``255``.
    top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
    the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
    data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self, mode, weight=None, pos_weight=None, ignore_index=255):
        super(BinaryCrossEntropyLoss, self).__init__(mode)
        self.weight = weight
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

        if self.pos_weight is not None:
            if isinstance(self.pos_weight, str):
                if self.pos_weight != 'dynamic':
                    raise ValueError(
                        "if type of `pos_weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.pos_weight))
            elif isinstance(self.pos_weight, (list, tuple)):
                self.pos_weight = paddle.to_tensor(np.array(self.pos_weight), dtype='float32')

        if self.weight is not None:
            if isinstance(self.weight, (list, tuple)):
                self.weight = paddle.to_tensor(np.array(self.weight), dtype='float32')

    def __clas__(self, logit, label, info=None):
        """
        Forward computation for clas
        Args:
        logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes
        label (Tensor): Label tensor, the data type is float32. Shape is (N, C), where each
                value is 0 or 1.
        info   (string):  Auxiliary information for loss calc.
        """
        if isinstance(self.pos_weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'), axis=0) # N, C -> C
            neg_num = paddle.sum(neg_index.astype('float32'), axis=0) # N, C -> C
            sum_num = pos_num + neg_num
            pos_weight = 2 * neg_num / (sum_num + 1e-7)
        else:
            pos_weight = self.pos_weight

        ignore_mask = (label != self.ignore_index).astype("float32")

        cost = F.binary_cross_entropy_with_logits(logit=logit, label=label, weight=self.weight,
                                                    reduction="none", pos_weight=pos_weight)

        cost = (ignore_mask * cost).sum() / (ignore_mask.sum() + 1e-5)
        return cost

     



                

