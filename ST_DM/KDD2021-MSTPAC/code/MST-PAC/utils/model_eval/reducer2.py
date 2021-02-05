#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the reducer2.py
 Author: map(map@baidu.com)
 Date: 2019/07/10 17:27:57
"""
import sys
import os
import sets
import math

def reducer():
    """
    todo
    """
    all_nixu_count = 0
    all_all_count = 0

    for line in sys.stdin:
        items = line.strip().split('\t')
        if len(items) != 2:
            continue
        (nixu_count, all_count) = items

        all_nixu_count += float(nixu_count)
        all_all_count += float(all_count)
    
    print "%s\t%s\t%s" % (all_nixu_count, all_all_count, all_nixu_count / all_all_count)
    print(all_nixu_count / all_all_count)
    return 0

if __name__ == "__main__":
    reducer()
