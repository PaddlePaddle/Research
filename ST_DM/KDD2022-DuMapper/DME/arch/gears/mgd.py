#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: mgd.py
func: 多全局描述head
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""

import paddle
import paddle.nn as nn

from mmflib.arch.gears.pooling import MAC, SPoC, GeM, GeMmp
from mmflib.arch.gears.neck import BottleNeck


class MultiGlobalDes(nn.Layer):
    """多全局描述
    """
    def __init__(self, input_dim, embd_shape=256, drop_rate=0.5):
        super(MultiGlobalDes, self).__init__()
        self.embd_shape = embd_shape
        self.mac = MAC()
        self.gem = GeMmp(p=3)
        self.spoc = SPoC()
        self.fc_mac = BottleNeck(input_dim, embd_shape, drop_rate)
        self.fc_spoc = BottleNeck(input_dim, embd_shape, drop_rate)
        self.fc_gem = BottleNeck(input_dim, embd_shape, drop_rate)

    def forward(self, fea_map):
        """forward
        Args:
            fea_map: feature map, dims:[batch, w, h, c]
        """     
        fea_mac = self.mac(fea_map)
        fea_spoc = self.spoc(fea_map)
        fea_gem = self.gem(fea_map)
        fea_mac = self.fc_mac(fea_mac.squeeze(axis=2).squeeze(axis=2))
        fea_gem = self.fc_gem(fea_gem.squeeze(axis=2).squeeze(axis=2))
        fea_spoc = self.fc_spoc(fea_spoc.squeeze(axis=2).squeeze(axis=2))
        return paddle.concat([fea_spoc, fea_mac, fea_gem], axis=-1)