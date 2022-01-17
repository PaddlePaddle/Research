#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: metric_loss.py
func: metric loss based on softmax
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/9
"""

from __future__ import print_function
from __future__ import division
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ArcMarginProduct(nn.Layer):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        self.fc = nn.Linear(in_features,
                            out_features,
                            weight_attr=weight_attr,
                            bias_attr=False)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = paddle.cast(x=(target > limit), dtype='float32')
        output = paddle.multiply(mask, x) + paddle.multiply((1.0 - mask), y)
        return output

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        weight = self.fc.weight
        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(weight), axis=0, keepdim=True))
        weight = paddle.divide(weight, weight_norm)
        
        cosine = paddle.matmul(input, weight)
        sine = paddle.sqrt((1.0 - paddle.square(cosine)).clip(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = self._paddle_where_more_than(cosine, 0, phi, cosine)       
        else:
            phi = self._paddle_where_more_than(cosine, self.th, phi, cosine - self.mm)
            

        one_hot = paddle.nn.functional.one_hot(label, self.out_features)
        one_hot = paddle.squeeze(one_hot, axis=[1])
        output = paddle.multiply(one_hot, phi) + paddle.multiply(
            (1.0 - one_hot), cosine)
        output = output * self.s

        return output, input_norm


if __name__ == "__main__":
    metric_head = ArcMarginProduct(256, 1000)
    inp = paddle.randn(shape=[100, 256])
    label = paddle.randint(high=999, shape=[100])
    y_pred, x_norm = metric_head(inp, label)
    print(y_pred.shape)
    print(x_norm.shape)