#!/usr/bin/python
# -*- coding=utf-8 -*-
"""
# @Author : Gao Can
# @Created Time : 2020-07-17 16:56:09
# @Description : 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def visualdl_log(metrics_output, train_loss, steps, phase):
    """log可视化，仅限于paddlecloud 平台任务
    """
    print("{phase} log: steps {steps}, loss {loss}, metrics: {metrics}".format(phase=phase, steps=steps, loss=train_loss, metrics=metrics_output))
    try:
        if metrics_output and len(metrics_output) != 0:
            import paddlecloud.visual_util as visualdl
            x_dic = {"x_name": "step", "x_value": steps}
            y_ls = []
            for key, value in metrics_output.items():
                y = {}
                y["y_name"] = key
                y["y_value"] = value
                y_ls.append(y)
            visualdl.show_fluid_trend(x_dic, y_ls, tag="train")
    except Exception:
        print("import paddlecloud.visual_util failed")


def print_eval_log(ret):
    prefix_log = "[%s evaluation] ave loss: %.4f," % (ret['phase'], ret['loss'])
    postfix_log = "data_num: %d, elapsed time: %.4f s" % (ret['data_num'], ret['used_time'])
    mid_log = " "
    for k, v in ret.items():
        if k not in ['phase', 'loss', 'data_num', 'used_time', 'key_eval']:
            mid_log = mid_log + "%s: %.4f, " % (k, round(v, 4))
    log = prefix_log + mid_log + postfix_log
    print(log)

def get_time():
    res = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    return res
