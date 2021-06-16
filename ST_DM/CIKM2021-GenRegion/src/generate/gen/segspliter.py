# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys
import gc

from region import geometry as mygeo
from region import error

class SegSpliter(object):
    """
    Desc: seg split
    """
    def __init__(self, segs, grid_size):
        """
        Desc: seg划分
        Args:
            self : self
            segs : 线段集合
        Return: 
            None 
        Raise: 
            None
        """
        self.segs = segs
        self.valids = [ True for seg in segs ]
        self.grid_size = grid_size
        self.grid_dict = None
        
    def __proc_seg_by_seg(self, i1, i2):
        """
        Desc: 计算两个seg之间的交点
        Args:
            self : self
            i1 : seg1 index
            i2 : seg2 index
        Return: 
            None 
        Raise: 
            None
        """
        seg1 = self.segs[i1]
        seg2 = self.segs[i2]
        ret = []
        points = seg1.intersect(seg2)
        if len(points) > 0:
            for pt in points:
                pt.trunc()
            segs = self.__split_seg_by_points(seg1, points)
            if len(segs) > 1:
                ret.extend(segs)
                self.valids[i1] = False
            segs = self.__split_seg_by_points(seg2, points)
            if len(segs) > 1:
                ret.extend(segs)
                self.valids[i2] = False
        return ret

    def __split_seg_by_points(self, seg, points):
        """
        Desc: 将seg按照这些points进行打断
        Args:
            self : self
            seg : 需要打断的原始seg
            points : 增加的点
        Return: 
            list of segs, 打断后的segs
        Raise: 
            None
        """
        pts = []
        line = seg.linestring() #正向反向两个linestrings
        for pt in points:
            if pt == seg.start or pt == seg.end:
                continue
            sppt = pt.point()
            dist = line.project(sppt) # 计算离有向线段起点的距离
            pts.append((pt, dist))
        pts.sort(key=lambda x: x[1])
        pts.append((seg.end, 0))
        last_point = seg.start
        ret = []
        for pt, dist in pts:
            if last_point == pt:
                continue
            newseg = mygeo.Segment(last_point, pt)
            ret.append(newseg)
            last_point = pt
        return ret

    def __add_dict(self, indexes):
        """
        Desc: 将segs添加self.grid_dict中
        Args:
            self : self
            indexes : list of seg index
        Return: 
            None 
        Raise: 
            None
        """
        for i in indexes:
            seg = self.segs[i]
            for grid in seg.grids(self.grid_size):
                if grid in self.grid_dict:
                    self.grid_dict[grid].append(i)
                else:
                    self.grid_dict[grid] = [i]

    def __get_intersect_segs(self, i1):
        """
        Desc: 计算seg与其它seg之间的交点, 产出新的seg
        Args:
            self : self
            i1 : seg1 index
        Return: 
            list of new segs
        Raise: 
            None
        """
        ret = []
        seg = self.segs[i1]
        for grid in seg.grids(self.grid_size):
            if grid in self.grid_dict:
                for i2 in self.grid_dict[grid]:
                    seg2 = self.segs[i2]
                    if self.valids[i2]:
                        segs = self.__proc_seg_by_seg(i1, i2)
                        if len(segs) > 0:
                            ret.extend(segs)
                            if not self.valids[i1]:
                                return ret
        return ret

    def run(self):
        """
        Desc: 进行线段打断, 过滤掉重复的seg
        Args:
            self : self
        Return: 
            list of new segs
        Raise: 
            None
        """
        self.grid_dict = {}
        i = 0
        while i < len(self.segs):
            if self.valids[i]:
                res_segs = self.__get_intersect_segs(i)
                if len(res_segs) > 0:
                    self.segs.extend(res_segs)
                    self.valids.extend([ True for rs in res_segs])
                if self.valids[i]:
                    self.__add_dict([i])
            i += 1
        ret = []
        seg_set = set([])
        for i in range(len(self.segs)):
            if self.valids[i]:
                seg = self.segs[i]
                seg_key = seg.askey()
                if seg_key not in seg_set:
                    seg_set.add(seg_key)
                    ret.append(seg)
        return ret
