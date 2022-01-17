#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: ce_loss.py
func: cross entrop loss
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/26
"""
import paddle.nn as nn
from paddle.nn import CrossEntropyLoss


class CELoss(nn.Layer):
    """cross entrop loss"""
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = CrossEntropyLoss()
    
    def forward(self, x, label):
        """"forward"""
        x = x['logits']
        loss = self.ce_loss(x, label)

        return loss
