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
Mixed loss
"""

import paddle
import paddle.nn.functional as F

from easymia.core.abstract_loss import AbstractLoss
from easymia.libs import manager

@manager.LOSSES.add_component
class MixedLoss(AbstractLoss):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.
    Args:
        losses (list[nn.Layer]): A list consisting of multiple loss classes
        coef (list[float|int]): Weighting coefficient of multiple loss
    Returns:
        A callable object of MixedLoss.
    """

    def __init__(self, mode, losses, coef):
        """
        init
        """
        super(MixedLoss, self).__init__(mode)
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.losses = losses
        self.coef = coef

    def __clas__(self, logits, labels, info=None):
        """
        分类
        """
        loss_list = []
        final_output = 0
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels, info)
            final_output += output * self.coef[i]
        return final_output

    def __seg__(self, logits, labels, info=None):
        """
        分割
        """
        loss_list = []
        final_output = 0
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels, info)
            final_output += output * self.coef[i]
        return final_output 