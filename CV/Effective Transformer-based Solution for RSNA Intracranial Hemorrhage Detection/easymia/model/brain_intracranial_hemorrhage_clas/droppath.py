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
Droppath, reimplement from https://github.com/yueatsprograms/Stochastic_Depth
"""

import numpy as np
import paddle
import paddle.nn as nn


class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        """
        forward
        """
        return self.drop_path(inputs)


#def main():
#    tmp = paddle.to_tensor(np.random.rand(8, 16, 8, 8), dtype='float32')
#    dp = DropPath(0.5)
#    out = dp(tmp)
#    print(out)
#
#if __name__ == "__main__":
#    main()