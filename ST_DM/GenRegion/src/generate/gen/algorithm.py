# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 

import gc
import sys

from region import geometry as mygeo
from region import error
from generate.gen import cluster
from generate.gen import generator
from generate.gen import regionalg
from generate.gen import segspliter


def rect_cluster(points, width=40):
    """
    Desc: 根据设定矩形大小的聚类算法, 每个类的MBR <= width * width
          采用层次聚类, 层次聚类选择合并的依据是 min (delta_x + delta_y)
          cluster A 的宽度为(xa, ya), cluster B 的宽度为(xb, yb)
          A, B合并后的类cluster C 的宽度为(xc, yc), xc < width, yc < width
              if xa + ya > xb + yb: 
                  delta_x = xc - xa
                  delta_y = yc - ya 
              else
                  delta_x = xc - xb
                  delta_y = yc - yb 
    Args:
        points : 需要聚类的点的集合, Point集合
        width : 类的宽度阈值
    Return: 
        list of cluster.Cluster
    Raises: 
        None
    """
    alg = cluster.HCAlgorithm(width)
    return alg.run(points)


def rect_classify(points, clusters, width=40):
    """
    Desc: 将点分类到相应的cluster中, 使得cluster的边界满足width阈值

    Args:
        points : 需要分类的Point集合

        clusters : list of cluster, cluster 可能会被修改, 增加一些新的point

        width : 类的宽度阈值
    Return: 
        [p3, p4, p5, ...], 无法分到cluster的所有点
    Raises: 
        None
    """
    alg = cluster.Classifier(clusters, width)
    return alg.run(points)


def region_filter(regions, grid_size=1024, thread_num=4):
    """
    Desc: 对原始regions进行过滤, 删除掉被包含的子区域, regions内部可能存在
          一个region A完全包含在另一个region B中, 我们将region A删除
    Args:
        regions : list of Region
        grid_size : 网格划分的大小
        thread_num : 线程数量
    Return: 
        删除后的list of Region, [r1, r2, r3, ...]
    Raises: 
        None
    """
    ret = None
    rf = regionalg.RegionFilter(regions)
    if thread_num <= 1:
        ret = rf.run(grid_size)
    else:
        ret = rf.multi_run(grid_size, thread_num)
    return ret


def split_segs_by_segs(segs, grid_size=1024):
    """
    Desc: 计算seg与seg之间的交点, 并将seg打断成多个不同的seg
          保证任意两个seg之间没有除首尾点之外的交点

    Args:
        segs : list of mygeo.Segment

        grid_size : 计算过程中的网格大小
    Return: 
        list of mygeo.Segment
    Raises: 
        None
    """
    ss = segspliter.SegSpliter(segs, grid_size)
    return ss.run()


def splitedsegs_2_regions(segs):
    """
    Desc: 将线段围绕组成的区域构建出一个region, 返回所有构建成功的region, 
          输入的segs中, 任意两个seg之间没有除首尾点之外的交点

    Args:
        segs : 线段集合
    Return: 
        new regions
    Raises: 
        None
    """
    rg = generator.RegionGenerator(segs)
    return rg.run()



def merge_regions(regions, raw_regions=None, grid_size=1024, \
                  area_thres=10000, width_thres=20):
    """
    Desc: region合并, 原始raw_region进行合并
          面积 < small_region_thres or area / length < small_width_thres 的region会被合并

    Args:
        regions : 合并前的region

        raw_regions : 原始region

        grid_size : 计算过程中网格划分的大小

        area_thres : region的最小面积, 一般10000

        width_thres : region的最小面积周长比, 一般10
    Return: 
        合并后的region list
    Raises: 
        None
    """
    rm = regionalg.RegionMerger(regions, raw_regions)
    segs = rm.run(grid_size, area_thres, width_thres)
    rg = generator.RegionGenerator(segs)
    return rg.run()


def __cluster_points(segments, regions, grid_size, clust_width):
    """
    Desc: 对points进行聚类, 包括segments points 和 regions points
    Args:
        segments : segment file or list mygeo.Segment
        regions : 原始已分割好的region list
        grid_size : 网格大小
        clust_width : 聚类宽度
    Return: 
        list of Cluster
    Raise: 
        None
    """
    if isinstance(segments, str):    # seg 文件传入
        error.debug("seg file: %s" % (segments))
        seg_points = generator.segments_to_cluster_points(mygeo.gen_segments(segments))
    else:
        error.debug("segment count: %d" % (len(segments)))
        seg_points = generator.segments_to_cluster_points(segments)
    error.debug("seg points count: %d" % (len(seg_points)))
    clusters = rect_cluster(seg_points, clust_width)
    del seg_points
    gc.collect()
    error.debug("cluster count: %d" % (len(clusters)))
    if regions is not None:
        reg_points = generator.regions_to_cluster_points(regions)
        error.debug("region to points count: %d" % (len(reg_points)))
        left_points = rect_classify(reg_points, clusters, clust_width * 2)
        error.debug("left points count: %d" % (len(left_points)))
        left_clusters = rect_cluster(left_points, clust_width)
        error.debug("left cluster count: %d" % (len(left_clusters)))
        clusters.extend(left_clusters)
        del reg_points
        del left_points
        del left_clusters
        error.debug("cluster count: %d" % (len(clusters)))
    return clusters

def generate_regions(segments, raw_regions=None, grid_size=1024, \
                     area_thres=10000, width_thres=20, \
                     simplify_thres=8, clust_width=20):
    """
    Desc: 根据用户指定的线段集合, 以及部分原始region信息, 对空间进行划分, 
          然后生成regions
    Args:
        segments : 线段集合, 这个必须要有, 否则没有意义
                   可以是list of mygeo.Segment, 也可以是文件名,
        raw_regions : 原始已分割好的Region list, 没有可以传入None
        area_thres : region的最小面积
        width_thres : region的最小宽度, region宽度 = 面积 / 周长
        simplify_thres : 区域简化阈值
        clust_width : 聚类的阈值, 多少范围内的点合并成一个点
    Return: 
        [] region list
    Raises: 
        None
    """
    regions = None
    if raw_regions is not None:
        regions = __filter_regions(segments, raw_regions, grid_size, clust_width)
    clusters = __cluster_points(segments, regions, grid_size, clust_width)
    pointmap = generator.clusters_to_pointmap(clusters)
    del clusters
    gc.collect()
    error.debug("pointmap finished")
    if isinstance(segments, str):
        segs = generator.simplify_by_pointmap(mygeo.gen_segments(segments), regions, pointmap)
    else:
        segs = generator.simplify_by_pointmap(segments, regions, pointmap)
    del pointmap
    gc.collect()
    error.debug("simply seg count: %d" % (len(segs)))
    segs = split_segs_by_segs(segs, grid_size)
    gc.collect()
    error.debug("split seg count: %d" % (len(segs)))
    regs = splitedsegs_2_regions(segs)
    del segs
    gc.collect()
    error.debug("origin region count: %d" % (len(regs)))
    regs = merge_regions(regs, regions, grid_size, area_thres, width_thres)
    gc.collect()
    error.debug("merge region count: %d" % (len(regs)))
    regs = region_filter(regs, grid_size)
    gc.collect()
    error.debug("filter region count: %d" % (len(regs)))
    return regs

