# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys
import datetime

from shapely import wkt

from region import Point
from region import Region
from generate.gen import algorithm
from conf import Conf

# common
def log(info):
    """
    Desc: 输出进度
    Args:
        info : 日志内容
    Return: 
        None 
    Raise: 
        None
    """
    print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info
    sys.stdout.flush()


class GenerateError(Exception):
    """
    Generate操作的错误
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


def wkt_2_regions(wkt_str):
    """
    Desc: 原始行政区划的wkt格式的字符串转换成单个polygon的wkt
    Args:
        wkt_str: 原始wkt字符串
    Returns: 
        list of Region
    Raises: 
        None
    """
    regions = []
    try:
        geo = wkt.loads(wkt_str)
        if geo.geom_type == "Polygon":
            regions.extend(algorithm.region_valid(Region(geo)))
        elif geo.geom_type == "MultiPolygon":
            for poly in geo: # Maybe we can replace this by using list(geo)
                regions.extend(algorithm.region_valid(Region(poly)))
        else:
            raise GenerateError("unknown wkt type:" + wkt_str)
    except Exception as e:
        raise GenerateError("wkt str to polygons failed:" + e.message)
    return regions


class RegionInfo(object):
    """
    Desc: 一个region对象
    """
    def __init__(self):
        """
        Desc: Admin对象的初始化 
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.level = None
        self.id = None
        self.name = None
        self.simple_name = None
        self.regions = None
        self.center = None
        self.region = None

    def compute_center(self):
        """
        Desc: 计算region的center, 如果一个region有多个regions，
              选择面积最大的region的中心店
        Args:
            self : self
        Return: 
            Point
        Raise: 
            None
        """
        if self.center is None:
            max_area = 0.0
            max_center = Point(0.0, 0.0)
            for region in self.regions:
                area = region.area()
                if area > max_area:
                    max_center = region.center()
                    max_area = area
            self.center = max_center


def output_regions(region_infos, file):
    """
    Desc: 输出region_infos
    Args:
        region_infos : 需要输出的region infos
        file :  输出的文件
    Return: 
        None 
    Raise: 
        None
    """
    with open(file, "w") as fo:
        for region_info in region_infos:
            fo.write("%d\t%s\t%.2f %.2f\t%s\t%s\t%d\n" % (region_info.id, region_info.name, \
                    region_info.center.x, region_info.center.y, str(region_info.region), \
                    region_info.simple_name, region_info.level))

