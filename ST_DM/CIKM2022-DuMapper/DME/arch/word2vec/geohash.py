#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: geohash.py
func: 对坐标进行geohash编码
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""
import time

import paddle
import numpy as np

import mmflib.arch.utils.ll_2_mc as cotool

class GeoHash(object):
    """geohash 编码
    将经纬度转为geo hash编码
    精度和编码位数对照表参见https://segmentfault.com/a/1190000023061274
    Args:
        precious:编码精度
    """
    def __init__(self, precious=40, coor_type="bd09ll"):
        assert coor_type in ["bd09ll", "bd09mc"]
        self.precious = precious
        self.coor_type = coor_type

    def __call__(self, x):
        """call"""
        batch_code = []
        batch = x.shape[0]
        for i in range(batch):
            lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
            j = 0
            code = []
            lon, lat = x[i][0], x[i][1]
            if self.coor_type == "bd09mc":
                lon, lat = cotool.Mercator2LonLat(lon, lat)
            while len(code) < self.precious:
                j += 1
                lat_mid = sum(lat_range) / 2
                lon_mid = sum(lon_range) / 2
                #经度
                if lon <= lon_mid:
                    code.append(0)
                    lon_range[1] = lon_mid
                else:
                    code.append(1)
                    lon_range[0] = lon_mid
                #纬度
                if lat <= lat_mid:
                    code.append(0)
                    lat_range[1] = lat_mid
                else:
                    code.append(1)
                    lat_range[0] = lat_mid
            batch_code.append(code)

        return paddle.to_tensor(batch_code, dtype='float64')


if __name__ == '__main__':
    x = np.array([[12960745.7300, 4833486.6700]])
    geo_hash = GeoHash(40, coor_type="bd09mc")
    x = paddle.to_tensor(x)

    t1 = time.time()
    code = geo_hash(x)
    print(code)
    t2 = time.time()
    print(t2 - t1)






