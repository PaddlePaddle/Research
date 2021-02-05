#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the mrr_ndcg.py
 Author: map(map@baidu.com)
 Date: 2019/07/10 17:27:57
"""
import sys
import numpy as np
from scipy import stats

def mrr_at_k(rs, k):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        But not binary is also supported!
    >>> [0, 0, 1, 0]
    >>> mrr_at_k(rs, 3)
    0.33333
    >>> [[0, 0, 2, 0]
    >>> mrr_at_k(rs, 3)
    0.33333
    >>> [0, 0, 1, 0]
    >>> mrr_at_k(rs, 2)
    0.

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = np.asfarray(rs)[:k]
    non_zero_idx = np.asarray(rs).nonzero()[0]
    if non_zero_idx.size == 0: return 0
    else: return 1. / (non_zero_idx[0] + 1)

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        # add by dingshiqiang01
        elif method == 2:
            return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


# ERR (Expected Reciprocal Rank)
# NOTE: max_grade should be *2*
# (dingshiqiang01) max=1~10, max_grade = 4
def err(ranking, max=10, max_grade=2):
    """
    todo
    """
    if max is None:
        max = len(ranking)

    ranking = ranking[:min(len(ranking), max)]
    ranking = map(float, ranking)

    result = 0.0
    prob_step_down = 1.0

    for rank, rel in enumerate(ranking):
        rank += 1
        utility = (pow(2, rel) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility)

    return result


# 排序后第一个位置
def exc_at_1(ranking):
    """
    todo
    """
    if len(ranking) < 1: return 0
    if int(ranking[0]) >= 3: return 1
    else: return 0

# 排序后第一个位置
def per_at_1(ranking):
    """
    todo
    """
    if len(ranking) < 1: return 0
    if int(ranking[0]) >= 4: return 1
    else: return 0

# 配对样本的双边t检验
# excel里面有个函数，TTEST（arr1, arr2, 2,1）
# 显著性校验 ttest, 输入两个浮点数数组
def ttest(arr1, arr2):
    """
    todo
    """
    return stats.ttest_rel(arr1, arr2)


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
        sum_satisfy += ndcg_at_k(label_list, k)
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
        sum_satisfy += mrr_at_k(label_list, k)
        query_num += 1.0
    return sum_satisfy / query_num

if __name__ == '__main__':
    content = {}
    for line in open(sys.argv[1]):
        line = line.strip("\r\n").split("\t")
        if len(line) < 3:
            continue
        qid = line[0]
        if qid not in content:
            content[qid] = []
        content[qid].append(list(map(float, line[1:])))
    k = 1
    ndcg = calc_ndcg_k(content, k)
    print("ndcg@1:", ndcg)
    k = 3
    ndcg = calc_ndcg_k(content, k)
    print("ndcg@3:", ndcg)
    k = 5
    ndcg = calc_ndcg_k(content, k)
    print("ndcg@5:", ndcg)
    k = 1
    mrr = calc_mrr_k(content, k)
    print("mrr@1:", mrr)
    k = 3
    mrr = calc_mrr_k(content, k)
    print("mrr@3:", mrr)
    k = 5
    mrr = calc_mrr_k(content, k)
    print("mrr@5:", mrr)
