#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: ini_parser.py
Author: tuku(tuku@baidu.com)
Date: 2019/06/22 23:06:02
"""

from __future__ import print_function

import sys
import os.path

from utils import load_conf_file 
from utils import flags

flags.DEFINE_string("conf_file", "./conf/distributed_train.conf", "The conf file path")
flags.DEFINE_string("sec_name", "DEFAULT", "The section name of conf item")
flags.DEFINE_string("conf_name", "", "The name of conf item")
flags.DEFINE_string("operation", "read", "read or write")
flags.DEFINE_string("new_value", "", "if operation is write, then set this value")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    if not os.path.isfile(FLAGS.conf_file):
        print("Conf file path[" + FLAGS.conf_file + "] does not exist!", file=sys.stderr)
        sys.exit(1)

    if not FLAGS.sec_name or len(FLAGS.sec_name) == 0:
        print("Invalid section name", file=sys.stderr)
        sys.exit(1)

    if not FLAGS.conf_name or len(FLAGS.conf_name) == 0:
        print("Invalid conf name", file=sys.stderr)
        sys.exit(1)

    if not FLAGS.operation or (FLAGS.operation != 'read' and FLAGS.operation != 'write'):
        print("Invalid operation", file=sys.stderr)
        sys.exit(1)

    if FLAGS.operation == "read":
        conf_parser = load_conf_file.get_config_parser()
        conf_parser.read(FLAGS.conf_file)
        for conf_name in FLAGS.conf_name.split('|'):
            if conf_parser.has_option(FLAGS.sec_name, conf_name):
                print(conf_parser.get(FLAGS.sec_name, conf_name))
                sys.exit(0)
            elif conf_parser.has_option("DEFAULT", conf_name):
                print(conf_parser.get("DEFAULT", conf_name))
                sys.exit(0)
            else:
                continue
        sys.exit(1)
    else:
        origin_file = open(FLAGS.conf_file)
        new_file_content = ""
        section_name = ""
        for origin_line in origin_file.readlines():
            line = origin_line.strip()
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1]
            if section_name == FLAGS.sec_name:
                params = line.split(":", 1)
                if len(params) == 2:
                    conf_name = params[0].strip()
                    value = params[1].strip()
                    if conf_name == FLAGS.conf_name:
                        origin_line = conf_name + ": " + FLAGS.new_value + "\n"
            new_file_content += origin_line
        origin_file.close()
        origin_file = open(FLAGS.conf_file, "w")
        origin_file.write(new_file_content)
        origin_file.close()
