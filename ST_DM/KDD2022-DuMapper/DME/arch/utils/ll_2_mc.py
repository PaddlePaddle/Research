#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: ll_2_mc.py
func: 墨卡托与经纬度间相互转换
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""
import math

SCALE_S = 20037508.34

def lonLat2Mercator(x, y):
    """Convert longitude/latitude to Mercator coordinate"""
    mx = x * SCALE_S / 180.
    my = math.log(math.tan((90. + y) * math.pi / 360.)) / (math.pi / 180.)
    my = y * SCALE_S / 180.
    
    return mx, my

def Mercator2LonLat(x, y):
    """Convert Mercotor point to longitude/latitude cooridinat"""
    lx = x / SCALE_S * 180.
    ly = y / SCALE_S * 180.
    ly = 180 / math.pi * (2 * math.atan(math.exp(ly * math.pi / 180.)) - math.pi / 2)
    
    return lx, ly


if __name__ == '__main__':
    x, y = 12962922.3800, 4832335.0200
    lx, ly = Mercator2LonLat(x, y)

    print(lx, ly)

    # lx, ly = bd09mc_to_bd09ll(x, y)
    # print(lx, ly)
