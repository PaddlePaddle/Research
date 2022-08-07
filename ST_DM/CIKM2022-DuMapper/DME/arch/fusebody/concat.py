# -*- coding: utf-8 -*
#!/usr/bin/env python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: concat.py
func: 多模态特征concat融合
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""
import paddle
import paddle.nn as nn


class ConcatFuse(nn.Layer):
    """concat fuse"""
    def __init__(self):
        super(ConcatFuse, self).__init__()

    def forward(self, img, word):
        """forward
        """
        x = paddle.concat([img, word], axis=-1)
        return paddle.cast(x, 'float32')


if __name__ == "__main__":
    x = paddle.randn(shape=[2,3])
    paddle.cast(x, 'float32')