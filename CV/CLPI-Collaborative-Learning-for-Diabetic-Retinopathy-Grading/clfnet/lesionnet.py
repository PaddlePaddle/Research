# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as FL
from paddle.fluid.param_attr import ParamAttr

__all__ = ["LesionNet"]


class LesionNet():
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            # channel, filter, stride, padding
            self.configs = [
                        [32, 5, 1, 2],  # 1024, 1024, 32 
                        [32, 2, 2, 0],  # 512,  512,  32
                        [64, 2, 2, 0],  # 256,  256,  64
                        [128, 2, 2, 0], # 128,  128,  128
                        [256, 2, 2, 0], # 64,   64,   256
                        [512, 2, 2, 0], # 32,   32,   512
                        [1024, 2, 2, 0],# 16,   16,   1024
                        [1024, 1, 1, 0],# 16,   16,   1024
                        [1024, 1, 1, 0],# 16,   16,   1024
                        # [class_dim, 1, 1, 0],   # 16,   16,   class_dim
                      ]
        else:
            self.configs = configs

    def mish(self, x):
        return x * FL.tanh(FL.softplus(x))

    def net(self, input, class_dim):
        """Create second stage model

         Args:
         class_dim: dim of multi-class vector

         Returns:
         A list contain 5 tensors / ops:
         - loss, cross-entropy loss tensor
         - accuracy, accuracy metric tensor
         - predict, model output tensor activated by softmax
         - reader, reader op to feed data into placeholder
         - hacked_img_id, img_id tensor
         """
        x = input
        for k, c in enumerate(self.configs):
            x = self.conv_bn_layer(x, *c, name="s1_layer{}".format(k))
            # x = FL.leaky_relu(x, 0.5)
            x = self.mish(x)

        logit = self.conv_layer(x, *[class_dim, 1, 1, 0], name="s1_layer{}".format(k + 1)) 
        predict = FL.softmax(logit, axis=1) # bs, 4, 1, 1 or bs, 4, 16, 16

        return predict

    def conv_layer(self,
                  input, 
                  num_filters,
                  filter_size,
                  stride,
                  padding,
                  name=None):
        """Create conv layer"""
        conv = FL.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=1,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1')
        return conv

    def conv_bn_layer(self,
                  input, 
                  num_filters,
                  filter_size,
                  stride,
                  padding,
                  name=None):
        """Create conv+bn layer"""
        conv = FL.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=1,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1')
        
        bn_name = name + ".bn"
        return FL.batch_norm(input=conv, 
                                       act=None,
                                       name=bn_name+'.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance',)