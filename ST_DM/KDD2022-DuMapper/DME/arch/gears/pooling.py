#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: pooling.py
func: 用于特征描述的池化操作 refer https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/06/15
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MAC(nn.Layer):
    """最大全局描述
    """
    def __init__(self):
        super(MAC, self).__init__()
    
    def forward(self, x):
        """forward"""
        return F.max_pool2d(x, (x.shape[-2], x.shape[-1]))


class SPoC(nn.Layer):
    """平均全局描述
    """
    def __init__(self):
        super(SPoC, self).__init__()
    
    def forward(self, x):
        """forward"""
        return F.avg_pool2d(x, (x.shape[-2], x.shape[-1]))


class GeM(nn.Layer):
    """gem全局描述
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        #self.p = Parameter(torch.ones(1)*p)
        self.p = p
        self.eps = eps
    
    def forward(self, x, p=3, eps=1e-6):
        """forward"""
        return F.avg_pool2d(x.clip(min=self.eps).pow(self.p), (x.shape[-2], x.shape[-1])).pow(1. / self.p)


class GeMmp(nn.Layer):
    """动态gem全局描述
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeMmp, self).__init__()
        self.p = self.create_parameter(shape=(2048,))
        self.p.set_value(np.ones(2048).astype("float32") * p)
        self.eps = eps
    
    def forward(self, x):
        """forward"""
        p_unsqueeze = self.p.unsqueeze(-1).unsqueeze(-1)
        return F.avg_pool2d(x.clip(min=self.eps).pow(p_unsqueeze), (x.shape[-2], x.shape[-1])).pow(1. / p_unsqueeze)

if __name__ == "__main__":
    x = paddle.randn((10, 2048, 14, 14))
    gem_pool = GeMmp()
    x_ = gem_pool(x)
    print(gem_pool.p)
    print(x_.shape)

