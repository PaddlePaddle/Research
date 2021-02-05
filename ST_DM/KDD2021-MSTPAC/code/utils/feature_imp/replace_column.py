#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: replace_column.py
Author: map(map@baidu.com)
Date: 2020/11/04 23:19:53
"""

import os
import sys
import random
from ConfigParser import SafeConfigParser

class ReplaceColumn():
    """
        repalce column
    """
    def __init__(self, origin_file, conf_path, new_data_path):
        """
            init
        """
        self.origin_file = origin_file
        self.new_data_path = new_data_path
        parser = SafeConfigParser()
        parser.read([conf_path])
        single_info = parser.get('common', 'single').strip().split(';')
        self.single = []
        for info in single_info:
            #[begin, end)
            begin_end = info.split('-')
            if len(begin_end) == 2:
                self.single.append(map(int, begin_end))
        self.union = []
        union_info = parser.get('common', 'union').strip().split(';')
        for info in union_info:
            union_list = info.split(',')
            if len(union_list) > 1:
                self.union.append(map(int, union_list))


    def read_file(self):
        """
            read file to fill content content_bak val_set
        """
        self.content = []
        self.all_val_set = []
        for line in file(self.origin_file):
            ll = line.strip("\r\n").split("\t")
            self.content.append(ll)
            line_len = len(ll)
            if len(self.all_val_set) == 0:
                self.all_val_set = [set([]) for i in range(line_len)]
            for i in range(line_len):
                self.all_val_set[i].add(ll[i])
        return 0

    def shuffle_print(self):
        """
            shuffle and print to file
        """
        size = len(self.content)
        random_index = range(size)
        for begin_end in self.single:
            for idx in range(begin_end[0], begin_end[1]):
                fp = open(self.new_data_path + '/' + str(idx), 'w+')
                random.shuffle(random_index)
                val_set = self.all_val_set[idx]
                for i in range(size):
                    org_val = self.content[i][idx]
                    new_val = self.content[random_index[i]][idx]
                    if org_val == new_val:
                        new_val = random.sample(val_set, 1)[0]
                    self.content[i][idx] = new_val
                    print >> fp, "\t".join(self.content[i])
                    #print "\t".join(self.content[i])
                    self.content[i][idx] = org_val
                fp.close()
        for idxs in self.union:
            fp = open(self.new_data_path + '/' + '_'.join(map(str, idxs)), 'w+')
            random.shuffle(random_index)
            for i in range(size):
                org_val = [self.content[i][idx] for idx in idxs]
                new_val = [self.content[random_index[i]][idx] for idx in idxs]
                self.replace(self.content[i], idxs, new_val)
                print >> fp, "\t".join(self.content[i])
                self.replace(self.content[i], idxs, org_val)
            fp.close()

    def replace(self, ll, idxs, vals):
        """
            replace ll with vals in idxs position
        """
        count = 0
        for idx in idxs:
            ll[idx] = vals[count]
            count += 1
        return 0


if __name__ == "__main__":
    replace_column = ReplaceColumn(sys.argv[1], sys.argv[2], sys.argv[3])
    replace_column.read_file()
    replace_column.shuffle_print()

