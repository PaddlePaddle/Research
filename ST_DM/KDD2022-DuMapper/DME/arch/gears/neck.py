
# coding=utf-8
#!/usr/bin/env python
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: neck.py
func: bottle neck
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/20
"""

import paddle.nn as nn

class BottleNeck(nn.Layer):
    """bottleneck"""
    def __init__(self, input_dim, num_bottleneck, droprate=0.5, inp_bn=False):
        """init"""
        super(BottleNeck, self).__init__()
        self.inp_bn = inp_bn
        add_block = []
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1D(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)] 
        add_block = nn.Sequential(*add_block)
        self.add_block = add_block
        if self.inp_bn:
            self.bn = nn.BatchNorm1D(input_dim)
            
    def forward(self, x):
        """forward"""
        if self.inp_bn:
            x = self.bn(x)
        x = self.add_block(x)
        return x