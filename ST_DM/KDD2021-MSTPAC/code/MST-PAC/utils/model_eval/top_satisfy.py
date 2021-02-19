#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: top_satisfy.py
Author: map(map@baidu.com)
Date: 2018/09/19 19:32:21
"""

import os
import sys

def take_second(info):
    """
    sort for elem 1
    """
    return info[1]

def take_first(info):
    """
    sort for elem 1
    """
    return info[0]

def top_n_satisfy2(content, n):
    """
    calc top_n satisfy for dataset
    """
    #print(n)
    sum_satisfy = 0.0
    query_num = 0.0
    for qid in content:
        label_sort = []
        score = []
        all_info = content[qid]
        num_label1 = 0
        for info in all_info:
            if info[0] > 0:
                num_label1 += 1
            label_sort.append([info[0], info[1]])
        label_sort.sort(key=take_second, reverse=True)
        satisfy = 0.0
        count = 0
        size = len(label_sort)
        for i in range(min(n, size)):
            cur_label = label_sort[i][0]
            if cur_label > 0:
                satisfy += 1
        cur_satisfy = satisfy / min(n, num_label1)
        sum_satisfy += cur_satisfy
        query_num += 1
    return sum_satisfy / query_num


def top_n_satisfy(content, n):
    """
    calc top_n satisfy for dataset
    """
    sum_satisfy = 0.0
    query_num = 0.0
    for qid in content:
        label_sort = []
        score = []
        all_info = content[qid]
        for info in all_info:
            label_sort.append([info[0], info[1]])
            score.append(info[1])
        label_sort.sort(key=take_first, reverse=True)
        score.sort(reverse=True)
        satisfy = 0.0
        count = 0
        size = len(label_sort)
        for i in range(size):
            cur_label = label_sort[i][0]
            cur_score = label_sort[i][1]
            if cur_label < 1:
                break
            if i >= n:
                break
            index = score.index(cur_score)
            count += 1
            if index < n:
                satisfy += 1
        if count == 0:
            sum_satisfy += 0.0
            query_num += 1
        else:
            sum_satisfy += satisfy / float(count)
            query_num += 1
    return sum_satisfy / query_num

content = {}
for line in open(sys.argv[1]):
    if len(line) < 1:
        continue
    if line[0] == "#":
        break
    line = line.strip("\r\n").split("\t")
    qid = line[0]
    if qid not in content:
        content[qid] = []
    content[qid].append(list(map(float, line[1:])))

n = 1
score = top_n_satisfy2(content, n)
print("top 1 satisfy:", score)

n = 3
score = top_n_satisfy2(content, n)
print("top 3 satisfy:", score)

n = 5
score = top_n_satisfy2(content, n)
print("top 5 satisfy:", score)
