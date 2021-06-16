# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
import sys
import datetime

def debug(info):
    """
    Desc: 打印debug信息 
    Args:
        str : 信息内容
    Return: 
        None 
    Raise: 
        None
    """
    print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info
    sys.stdout.flush()

class RegionError(Exception):
    """
    Region操作异常, 无法继续工作
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)

