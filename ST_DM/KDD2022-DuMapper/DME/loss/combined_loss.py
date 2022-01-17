#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: combined_loss.py
func: 对不同的loss进行加权组合
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/22
"""

import paddle
import paddle.nn as nn

from mmflib.loss.ce_loss import CELoss


class CombinedLoss(nn.Layer):
    """组合不同loss
    """
    def __init__(self, config_list):
        super(CombinedLoss, self).__init__()
        self.loss_func = []
        self.loss_name = []
        self.loss_weight = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
            
        for config in config_list:
            print(config)
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))
            self.loss_name.append(name)

    def __call__(self, input, batch):
        """call"""
        loss_dict = {}
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch)
            weight = self.loss_weight[idx]
            key = self.loss_name[idx]
            loss = {key: loss * weight}
            loss_dict.update(loss)
        loss_dict["loss"] = sum(list(loss_dict.values()))
        return loss_dict

