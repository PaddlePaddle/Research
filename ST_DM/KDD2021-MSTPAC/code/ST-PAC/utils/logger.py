#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

"""
logger utilities.
"""

import time
import sys
import re
import os
import six
import numpy as np
import logging
import logging.handlers

"""
******functions for string processing******
"""


def pattern_match(pattern, line):
    """
    Check whether a string is matched
    Args:
      pattern: mathing pattern
      line : input string
    Returns:
      True/False
    """
    if re.match(pattern, line):
        return True
    else:
        return False


"""
******functions for parameter processing******
"""


def print_progress(task_name, percentage, style=0):
    """
    Print progress bar
    Args:
      task_name: The name of the current task
      percentage: Current progress
      style: Progress bar form
    """
    styles = ['#', '█']
    mark = styles[style] * percentage
    mark += ' ' * (100 - percentage)
    status = '%d%%' % percentage if percentage < 100 else 'Finished'
    sys.stdout.write('%+20s [%s] %s\r' % (task_name, mark, status))
    sys.stdout.flush()
    time.sleep(0.002)


def display_args(name, args):
    """
    Print parameter information
    Args:
      name: logger instance name
      args: Input parameter dictionary
    """
    logger = logging.getLogger(name)
    logger.info("The arguments passed by command line is :")
    for k, v in sorted(v for v in vars(args).items()):
        logger.info("{}:\t{}".format(k, v))


def import_class(module_path, module_name, class_name):
    """
    Load class dynamically
    Args:
      module_path: The current path of the module
      module_name: The module name
      class_name: The name of class in the import module
    Return:
      Return the attribute value of the class object
    """
    if module_path:
        sys.path.append(module_path)
    module = __import__(module_name)
    return getattr(module, class_name)


def str2bool(v):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Argument Class
    """

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        Add argument
        """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def init_log(
        log_path,
        level=logging.INFO,
        when="D",
        backup=7,
        format="%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d * %(message)s",
        datefmt=None):
    """
    init_log - initialize log module

    Args:
      log_path      - Log file path prefix.
                      Log data will go to two files: log_path.log and log_path.log.wf
                      Any non-exist parent directories will be created automatically
      level         - msg above the level will be displayed
                      DEBUG < INFO < WARNING < ERROR < CRITICAL
                      the default value is logging.INFO
      when          - how to split the log file by time interval
                      'S' : Seconds
                      'M' : Minutes
                      'H' : Hours
                      'D' : Days
                      'W' : Week day
                      default value: 'D'
      format        - format of the log
                      default format:
                      %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                      INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
      backup        - how many backup file to keep
                      default value: 7

    Raises:
        OSError: fail to create log directories
        IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if len(logger.handlers) > 0:
        logger.handlers = []

    # console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
                "%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d * %(message)s", None))
    logger.addHandler(console_handler)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log", when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log.wf", when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_level(level):
    """
    Reak-time set log level
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.info('log level is set to : %d' % level)


def get_level():
    """
    get Real-time log level
    """
    logger = logging.getLogger()
    return logger.level

