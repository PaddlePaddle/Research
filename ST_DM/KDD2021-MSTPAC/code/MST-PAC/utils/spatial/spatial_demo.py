#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: spatial_demo.py
Author: map(zhuoan@baidu.com)
Date: 2018/10/12 13:35:42
"""

import os
import sys
import time

import spatial_pyso as spatial

base32_codes = [
            '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def _get_bin(dec_num):
    b = [0] * 5 #dec_num <= 31 
    s = dec_num
    i = 4
    while s > 0:  
        b[i] = s % 2
        s = s // 2
        i -= 1
    return b

geohash_id = {}
for i in range(len(base32_codes)):
    geohash_id[base32_codes[i]] = _get_bin(i)


def _geohash_to_bits(geohash):
    """
        w4gx223 -> 00001 11001 01101 ...
    """
    if len(geohash) < 1:
        return [0] * 40
    bits = []
    for c in geohash:
        bits.extend(geohash_id[c])
    return bits


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('gb18030')
    
    begin_time = time.time()
    sample_num = 0
    for line in sys.stdin:
        sample_num += 1
        ll = line.strip('\r\n').split(';') # 12012 3232;12l32 323;1234 232
        points = []
        for p in ll:
            points.append(tuple(map(float, p.split())))
        for p in points:
            geohash = spatial.calc_mc_geohash(p[0], p[1])
            print("%s->geo_hash:%s, %s" % (p, geohash, "".join(map(str, _geohash_to_bits(geohash)))))
            longi_lati = spatial.mc_to_ll(p[0], p[1]) 
            print("%s->mc_to_ll:%s" % (p, longi_lati))
            x_y = spatial.ll_to_mc(longi_lati[0], longi_lati[1]) 
            print("%s->ll_to_mc:%s" % (longi_lati, x_y))
        # 预测必须使用以下格式
        dis_results = spatial.calc_poi_dis(points)
        print("%s->%s" % (points, ";".join(map(str, dis_results))))

    end_time = time.time()
    spend_time = end_time - begin_time
    sys.stderr.write("spend_time:%s\tsample_num:%s\tavg_time:%s\n" % (spend_time, sample_num, \
            spend_time * 1. / sample_num))

