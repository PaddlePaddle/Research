#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""wrapper of paddle executor
"""

import sys
import os
import traceback
import logging

from paddle import fluid

class Executor(object):

    """Simple Executor. """

    def __init__(self, place='cpu'):
        """init of class """
        super(Executor, self).__init__()

        if place == 'cpu':
            self.exe = fluid.Executor(fluid.CPUPlace())
        else:
            raise ValueError('currently only support cpu place')
        self.exe.run(fluid.default_startup_program())

    def run(self, *args, **kwargs):
        """wrapper of Executor.run()

        Args:
            *args (TYPE): NULL
            **kwargs (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        return self.exe.run(fluid.default_main_program(), *args, **kwargs)


if __name__ == "__main__":
    """run some simple test cases"""
    pass

