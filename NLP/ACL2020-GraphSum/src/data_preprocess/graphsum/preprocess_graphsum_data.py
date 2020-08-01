#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import time
import os
from data_preprocess.graphsum import data_builder_graphsum


def do_format_to_json(args):
    """format dataset to json format"""
    print(time.clock())
    data_builder_graphsum.format_to_json(args)
    print(time.clock())


def do_format_to_paddle(args):
    """format dataset for paddle models"""
    print(time.clock())
    data_builder_graphsum.format_to_paddle(args)
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
    parser.add_argument("-data_path", default='MultiNews_data_tfidf_paddle')
    parser.add_argument("-vocab_path", default='../../vocab/spm9998_3.model')

    parser.add_argument("-train_src", default='multi-news-processed/train.txt.src')
    parser.add_argument("-train_tgt", default='multi-news-processed/train.txt.tgt')
    parser.add_argument("-valid_src", default='multi-news-processed/val.txt.src')
    parser.add_argument("-valid_tgt", default='multi-news-processed/val.txt.tgt')
    parser.add_argument("-test_src", default='multi-news-processed/test.txt.src')
    parser.add_argument("-test_tgt", default='multi-news-processed/test.txt.tgt')

    parser.add_argument("-shard_size", default=4000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=10, type=int)
    parser.add_argument('-max_src_ntokens', default=100, type=int)
    parser.add_argument('-sim_threshold', default=0.05, type=float)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-dataset', default='', help='train, valid or test, default will process all datasets')
    parser.add_argument('-n_cpus', default=1, type=int)

    parser.add_argument("-sim_function", default='tf-idf', type=str, help='tf-idf or lsi or lda')
    parser.add_argument('-num_topics', default=20, type=int)
    parser.add_argument("-find_opt_num", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        os.mkdir(args.json_path)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    do_format_to_json(args)
    do_format_to_paddle(args)


