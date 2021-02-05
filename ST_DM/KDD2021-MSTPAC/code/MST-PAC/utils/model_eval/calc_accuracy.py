#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: calc_accuracy.py
Author: tuku(tuku@baidu.com)
Date: 2018/04/11 14:00:32
"""

import sys
import os

def accuracy_score():
    """
        metric for sample classify score, if label == score, then plus 1 else 0
    """
    good = 0
    all = 0
    for line in sys.stdin:
        items = line.strip().split('\t')
        if len(items) != 3:
            continue
        qid, label, score = items
        if int(label) == int(score):
            good += 1
        all += 1
    if all > 0:
        print(float(good) / all)
    else:
        print(0.0)
    return 0

if __name__ == "__main__":
    accuracy_score()
