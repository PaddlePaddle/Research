# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################
"""
日志记录器
"""

import sys
import time

import paddle

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2


def log(level=2, message=""):
    """
    log
    """
    if paddle.distributed.ParallelEnv().local_rank == 0:
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if log_level >= level:
            print(
                "{} [{}]\t{}".format(current_time, levels[level],
                                     message).encode("utf-8").decode("latin1"))
            sys.stdout.flush()


def debug(message=""):
    """
    debug
    """
    log(level=3, message=message)


def info(message=""):
    """
    info
    """
    log(level=2, message=message)


def warning(message=""):
    """
    warning
    """
    log(level=1, message=message)


def error(message=""):
    """
    error
    """
    log(level=0, message=message)


class Recorder(object):
    """
    日志记录器
    """
    def __init__(self, keys):
        """
        keys: list[str]
        """

        self.__states = {k: [] for k in keys}
    
    def clear(self, key):
        """
        清空
        """
        if key in self.__states.keys():
            self.__states[key] = []
        else:
            raise ValueError("Check wheather {} in initial keys.".format(key))

    def record(self, key, value):
        """
        记录
        """
        if key in self.__states.keys():
            self.__states[key].append(value)
        else:
            raise ValueError("Check wheather {} in initial keys.".format(key))
    
    def get(self, key, reduction=None):
        """
        获取
        
        reduction: str, ['sum', 'mean', None]
        """
        if key in self.__states.keys():
            if reduction == "sum": 
                return sum(self.__states[key])
            elif reduction == "mean":
                return sum(self.__states[key]) / len(self.__states[key])
            elif reduction is None:
                return self.__states[key]
        else:
            raise ValueError("Check wheather {} in initial keys.".format(key))

if __name__ == '__main__':
    logger = Recorder(["r1", "r2", "r3"])
    logger.record("r1", 1)
    logger.record("r1", 2)
    print(logger.get("r1"))
    print(logger.get("r1", "mean"))