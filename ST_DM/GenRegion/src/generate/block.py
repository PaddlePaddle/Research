# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys
import gc
import threading

from region import load_regions
from region import line_simp
from region import dump_segments
from region import dump_regions
from region import Region
from region import Point
from region import Segment
from generate import etl
from generate.conf import Conf
from generate.gen import algorithm
from generate.gen import regionalg
from generate import roadnet_load


def __list_to_segs(point_list):
    """
    Desc: 将一个list的tuple坐标转换成segs
    Args:
        A list of tuples
    Return:
        A list of mygeo.Segment
    Raise:
        None
    """
    Point.precision = 8
    line = []
    for pt in point_list:
        x,y = pt
        line.append(Point(float(x), float(y)))
    line = line_simp(line, Conf.block_simp_threshold) 
    if len(line) <= 1:
        raise etl.GenerateError("line point count <= 1")
    segs = []
    lastpt = None
    for pt in line:
        if lastpt is not None and not lastpt == pt:
            segs.append(Segment(lastpt, pt))
        lastpt = pt
    return segs


# added by Ming
def __block_only_info(regions):
    """
    Desc: 将block的regions输出成region infos
    Args:
        regions: 新生成的block数据
    Return:
        list of etl.RegionInfo
    Raise:
        None
    """
    blocks = []
    id = 0
    for region in regions:
        reg_info = etl.RegionInfo()
        reg_info.level = Conf.block_level
        reg_info.id = id
        reg_info.name = ''
        reg_info.simple_name = ''
        reg_info.center = region.center()
        reg_info.region = region
        blocks.append(reg_info)
        id+=1
    return blocks


def run_city(city, input_path, output_path):
    etl.log(city)
    road = roadnet_load.Roadnet(input_path)
    point_lists = road.to_seglists()
    ret = []
    for edge in point_lists:
        segs = __list_to_segs(edge)
        ret.extend(segs)
    etl.log("roadnet seg count: %d" % (len(ret)))
    dump_segments(ret, "%s_segs.txt"%city)
    del ret
    Point.set_precision(0)
    etl.log('block gen')
    regs = algorithm.generate_regions("%s_segs.txt"%city, grid_size = Conf.grid_level3, \
                                      area_thres = Conf.min_block_area, \
                                      width_thres = Conf.min_block_width, \
                                      simplify_thres = Conf.block_simp_threshold, \
                                      clust_width = Conf.clust_width)
    etl.log("block info")
    blocks = __block_only_info(regs)
    etl.output_regions(blocks, output_path)
    etl.log("finished")

def run(city_name, in_file, out_file):
    run_city(city_name, in_file, out_file)
    
