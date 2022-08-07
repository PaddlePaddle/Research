#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: topk.py
func: top k 精度
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/22
"""
import paddle
import paddle.nn as nn

class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        metric_dict = dict()
        bath_size = label.shape[0]
        for k in self.topk:
            metric_dict["top{}".format(k)] = paddle.metric.accuracy(
                x, label.reshape([bath_size, -1]), k=k)
        return metric_dict


if __name__ == '__main__':
    import paddle
    import numpy as np
    x = {}
    x["logits"] = paddle.to_tensor(np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]), dtype='float32')
    target = paddle.to_tensor(np.array([1, 2, 3, 4])).reshape([4, -1])

    metric = TopkAcc()
    print(metric(x, target)['top1'].detach().cpu().numpy())