#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the calc_ndcg_mrr.py
 Author: map(map@baidu.com)
 Date: 2019/07/10 17:27:57
"""
import sys
import numpy as np
from scipy import stats
import mrr_ndcg

def take_elem(info):
    """
    sort for elem 1
    """
    return info[1]

def calc_ndcg_k(content, k):
    """
    calc ndcg at k for dataset
    """
    sum_satisfy = 0.0
    query_num = 0.0
    for qid in content:
        all_info = content[qid]
        label_sort = []
        label_list = []
        for info in all_info:
            label_sort.append([info[0], info[1]])
        label_sort.sort(key=take_elem, reverse=True)
        for label in label_sort:
            label_list.append(label[0])
        sum_satisfy += mrr_ndcg.ndcg_at_k(label_list, k)
        query_num += 1.0
    return sum_satisfy / query_num


def calc_mrr_k(content, k):
    """
    calc mrr at k for dataset
    """
    sum_satisfy = 0.0
    query_num = 0.0
    for qid in content:
        all_info = content[qid]
        label_sort = []
        label_list = []
        for info in all_info:
            label_sort.append([info[0], info[1]])
        label_sort.sort(key=take_elem, reverse=True)
        for label in label_sort:
            label_list.append(label[0])
        sum_satisfy += mrr_ndcg.mrr_at_k(label_list, k)
        query_num += 1.0
    return sum_satisfy / query_num

if __name__ == '__main__':
    content = {}
    for line in file(sys.argv[1]):
        line = line.strip("\r\n").split("\t")
        qid = line[0]
        if qid not in content:
            content[qid] = []
        content[qid].append(map(float, line[1:]))
    k = 1
    ndcg = calc_ndcg_k(content, k)
    print "top ndcg@1:", ndcg
    k = 3
    ndcg = calc_ndcg_k(content, k)
    print "top ndcg@3:", ndcg
    k = 5
    ndcg = calc_ndcg_k(content, k)
    print "top ndcg@5:", ndcg
    print "-------------"
    k = 1
    mrr = calc_mrr_k(content, k)
    print "top mrr@1:", mrr
    k = 3
    mrr = calc_mrr_k(content, k)
    print "top mrr@3:", mrr
    k = 5
    mrr = calc_mrr_k(content, k)
    print "top mrr@5:", mrr
