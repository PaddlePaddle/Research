#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: feature_sort.py
Author: map(map@baidu.com)
Date: 2020/11/05 20:46:55
"""

import os
import sys


def get_name(idx2name, info):
    """
        get name from idx2name
    """
    idx = info.strip().split('_')
    if len(idx) == 1:
        if idx[0] in idx2name:
            return idx2name[idx[0]]
        else:
            return idx[0]
    elif len(idx) > 1:
        ret = []
        for i in idx:
            if i in idx2name:
                ret.append(idx2name[i])
            else:
                ret.append(i)
        return '#'.join(ret)

idx2name = {}
for line in file(sys.argv[2]):
    ll = line.strip("\r\n").split("\t")
    idx2name[ll[0]] = ll[1]

all_score = {}
base = 0

for line in file(sys.argv[1]):
    idx, score = line.strip("\r\n").split("\t")
    if idx == 'base':
        base = float(score)
    else:
        all_score[idx] = base - float(score)

sort_score = sorted(all_score.iteritems(), key=lambda d:d[1], reverse = True)
for info in sort_score:
    print "%s\t%f" % (get_name(idx2name, info[0]), info[1])
