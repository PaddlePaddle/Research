"""
nixu
"""


import sys
import os
import sets
import math


def reducer():
    """
    :return:
    """
    qid_info = []
    pqid = ""
    
    for line in sys.stdin:
        items = line.strip().split('\t')
        if len(items) != 3:
            continue
        (qid, label, score) = items
        
        if qid != pqid and pqid != "":
            output(qid_info)
            qid_info = []

        qid_info.append([int(label), float(score)])
        pqid = qid
    
    if len(qid_info) > 0:
        output(qid_info)
        qid_info = []
    
    return 0


def output(qid_info):
    """
    :param qid_info:
    :return:
    """
    L = len(qid_info)
    qid_count = 0
    qid_score = 0.0
    for i in range(L):
        label_i = qid_info[i][0]
        score_i = qid_info[i][1]
        for j in range(i + 1, L):
            label_j = qid_info[j][0]
            score_j = qid_info[j][1]
            if label_i != label_j:
                qid_count += 1
                if (label_i < label_j and score_i > score_j) \
                        or (label_i > label_j and score_i < score_j):
                    qid_score += 1
                elif score_i == score_j:
                    qid_score += 0.5
    print "%s\t%s" % (str(qid_score), str(qid_count))
    return 0


if __name__ == "__main__":
    reducer()
