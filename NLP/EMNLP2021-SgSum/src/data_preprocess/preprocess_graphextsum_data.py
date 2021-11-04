#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#

"""
File: preprocess_graphsum_data.py
Author: chenmoye(chenmoye@baidu.com)
Date: 2021-10-18 14:50
Desc:
"""

import argparse
import time
import os
import data_builder_graphextsum


def do_format_to_json(args):
    """format dataset to json format"""
    print(time.clock())
    data_builder_graphextsum.format_to_json(args)
    print(time.clock())


def do_format_to_paddle(args):
    """format dataset for paddle models"""
    print(time.clock())
    data_builder_graphextsum.format_to_paddle(args)
    print(time.clock())


def str2bool(v):
    """For Bool args"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_to_json or format_to_paddle')
    parser.add_argument("-json_path", default='json_data/')
    parser.add_argument("-data_path", default='input_data')
    parser.add_argument("-vocab_path", default='config/vocab/spm9998_3.model')

    parser.add_argument("-roberta_vocab_file",
                        default='config/bpe/gpt2_bpe/vocab.txt',
                        type=str)
    parser.add_argument("-encoder_json_file",
                        default='config/bpe/gpt2_bpe/encoder.json',
                        type=str)
    parser.add_argument("-vocab_bpe_file",
                        default='config/bpe/gpt2_bpe/vocab.bpe',
                        type=str)

    parser.add_argument("-train_src", default='data/example.source')
    parser.add_argument("-train_tgt", default='data/example.target')
    parser.add_argument("-valid_src", default='data/example.source')
    parser.add_argument("-valid_tgt", default='data/example.target')
    parser.add_argument("-test_src", default='data/example.source')
    parser.add_argument("-test_tgt", default='data/example.target')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=5, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=768, type=int)
    parser.add_argument('-sim_threshold', default=0.1, type=float)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-dataset', default='', help='train, valid or test, default will process all datasets')
    parser.add_argument('-n_cpus', default=20, type=int)

    parser.add_argument("-sim_function", default='tf-idf', type=str, help='tf-idf or lsi or lda')
    parser.add_argument('-num_topics', default=20, type=int)
    parser.add_argument("-find_opt_num", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-process_wiki", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-summary_sent_num", type=int, default=30)
    parser.add_argument("-label_selection_type", default='greedy_selection', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        os.mkdir(args.json_path)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    do_format_to_json(args)
    do_format_to_paddle(args)


