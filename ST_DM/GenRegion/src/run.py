# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 


from generate import etl
from generate import block
import argparse
import os

parser = argparse.ArgumentParser(description='path')
parser.add_argument('--in-file', default='../data/Wuhan_Edgelist.csv', help='input file')
parser.add_argument('--out-file', default='../result/block_wuhan', help='output file')
parser.add_argument('--city-name', default='wuhan', help='city name')


args = parser.parse_args()
abs_path = os.path.split(os.path.realpath(__file__))[0]

out_file = os.path.join(abs_path, args.out_file) if args.out_file == '../result/block_wuhan' else args.out_file
in_file = os.path.join(abs_path, args.in_file) if args.in_file == \
                                                      '../data/Wuhan_Edgelist.csv' else args.in_file

block.run(args.city_name, in_file, out_file)

