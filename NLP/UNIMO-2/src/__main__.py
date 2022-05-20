# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m unimo2方式直接执行。

Authors: liwei85(liwei85@baidu.com)
Date:    2021/04/15 10:40:40
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from unimo2.cmdline import main
sys.exit(main())
