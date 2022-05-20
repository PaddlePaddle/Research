#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File: new_stat_res.py
Author: liwei(liwei85@baidu.com)
Date: 2021-08-15 15:34
Desc: eval results statistics
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
from args import ArgumentGroup
import codecs

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "stat", "stat configuration")
model_g.add_arg("log_dir", str, None, "stat log dir")
model_g.add_arg("score_name", str, "avg_recall@1", "random slot in log file")
model_g.add_arg("key_words", str, "job.log.0", "key words indentify log file")
model_g.add_arg("line_prefix", str, "[dev evaluation]", "key words indentify scores to stat")
model_g.add_arg("score_slot", int, -1, "score slot in stat line")
model_g.add_arg("final_res_file", str, "final_res.txt", "the file to save final stat score")

args = parser.parse_args()


def get_all_res(infile, score_name):
    scores = []
    step = 0
    with codecs.open(infile, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\r\n')

            if 'progress:' in line:
                step = int(line.split(', ')[2].split(': ')[1])
            elif line.startswith(args.line_prefix):
                res_list = line.split(', ')
                for res in res_list:
                    key, value = res.split(': ')
                    if key == score_name:
                        scores.append((step, float(value)))
                        break
    return scores


def print_stat(score_files):
    nums = len(score_files)
    max_score_file, (max_score_step, max_score) = score_files[0]
    min_score_file, (min_score_step, min_score) = score_files[-1]
    median_score_file, (median_score_step, median_score) = score_files[int(nums / 2)]
    mean_score = np.average([score for file, (step, score) in score_files])

    log = '%d\nmax_score %f max_file %s max_score_step %d\nmin_score %f min_file %s min_score_step %d\n' \
          'median_score %f median_file %s median_score_step %d\navg_score %f' % \
          (nums, max_score, max_score_file, max_score_step, min_score, min_score_file, min_score_step,
           median_score, median_score_file, median_score_step, mean_score)
    print(log)


score_file = {}
for file in os.listdir(args.log_dir):
    if args.key_words in file:
        scores = get_all_res(os.path.join(args.log_dir, file), args.score_name)
        score_file[file] = scores

best_score_file = []
for file, scores in score_file.items():
    sorted_scores = sorted(scores, key=lambda a: a[1], reverse=True)
    if len(sorted_scores) > 0:
        best_score_file.append((file, sorted_scores[0]))
    if len(sorted_scores) > 1:
        best_score_file.append((file, sorted_scores[1]))

best_score_file = sorted(best_score_file, key=lambda a: a[1][1], reverse=True)

sys.stdout = codecs.open(os.path.join(args.log_dir, args.final_res_file), 'w', encoding='utf-8')
print_stat(best_score_file)
