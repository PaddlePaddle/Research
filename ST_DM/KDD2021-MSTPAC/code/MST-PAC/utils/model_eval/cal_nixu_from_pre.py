#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the cal_nixu_from_pre.py
 Author: map(map@baidu.com)
 Date: 2019/07/10 17:27:57
"""
import sys
from itertools import groupby
from operator import itemgetter

if __name__ == "__main__":
    total_score = 0.
    total_count = 0
    group_count = 0
    count_of_num = {}
    score_of_num = {}
    pairs = []
    pre_qid = ""
    for line in sys.stdin:
        params = line.split("\t")
        qid = params[0]
        label = params[1]
        res = params[2]
        if qid != pre_qid and len(pairs) > 1:
            group_count += 1
            if group_count % 100000 == 0:
                sys.stdout.write('%cProcess %d group(s), current_metric: %.4f' % \
                    (13, group_count, total_score / total_count))
                sys.stdout.flush()
            qid_count = 0
            qid_score = 0
            L = len(pairs)
            for i in range(L):
                label_i = pairs[i][0]
                score_i = pairs[i][1]
                for j in range(i + 1, L):
                    label_j = pairs[j][0]
                    score_j = pairs[j][1]
                    if label_i != label_j:
                        qid_count += 1
                        # reverse ordered.
                        if (score_i < score_j and label_i > label_j) or (score_i > score_j and label_i < label_j):
                            qid_score += 1
            if L not in count_of_num:
                count_of_num[L] = 0
                score_of_num[L] = 0.0
            count_of_num[L] += qid_count
            score_of_num[L] += qid_score
            total_count += qid_count
            total_score += qid_score
            pairs = []
        pre_qid = qid
        pairs.append([int(label), float(res)])
        result = qid + "\t" + label + "\t" + res + "\n"
    sys.stdout.write('\n')
    print '%d group(s) processed.' % group_count
    print 'metrics: %.4f' % (total_score / total_count)
    for i in range(20):
        if i in count_of_num:
            print '---len %s metrics: %.4f' % (str(i), score_of_num[i] / count_of_num[i])
            print '---len %s score_num : %.3f' % (str(i), score_of_num[i])
            print '---len %s count num : %d' % (str(i), count_of_num[i])
    print 'score_sum: %.3f' % total_score
    print 'valid_pair_count: %d' % total_count

