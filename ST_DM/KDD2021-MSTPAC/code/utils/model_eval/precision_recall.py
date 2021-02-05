#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the precision_recall.py
 Author: map(map@baidu.com)
 Date: 2019/07/10 17:27:57
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class EvalHelper(object):
    """
    Eval
    """
    def __init__(self):
        """
        init
        """
        print("eval start...")

    def run_stream(self):
        """
        stream
        """
        all_cnt = [0.0, 0.0]
        pred_cnt = [0.0, 0.0]
        true_cnt = [0.0, 0.0]
        for line in sys.stdin:
            line = line.strip("\n").split("\t")
            if 3 != len(line):
                continue

            qid = line[0]
            label = int(line[1])
            pred = int(line[2])
            if pred == label:
                true_cnt[label] += 1

            pred_cnt[pred] += 1
            all_cnt[label] += 1

        return true_cnt[1] / pred_cnt[1], true_cnt[1] / all_cnt[1]

if __name__ == "__main__":

    eh = EvalHelper()
    p, r = eh.run_stream()

    sys.exit(0)
