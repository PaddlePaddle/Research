# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 

import math

from region import error
from region import geometry as mygeo
from generate.gen import cluster

def cos(invec, outvec):
    """
    Desc: 对于一个节点, invec为进入这个节点的向量, outvec为从这个节点退出的向量
          计算进入向量与退出向量之间夹角cos
    Args:
        invec : invec, (inx, iny)
        outvec : outvec, (outx, outy)
    Return: 
        从invec走向outvec的夹角, [-1,1]
    Raise: 
        None
    """
    (inx, iny) = invec
    (outx, outy) = outvec
    if (inx == 0 and iny == 0) or (outx == 0 and outy == 0):
        raise error.RegionError("vector is zero")
    inlen = math.sqrt(inx * inx + iny * iny)
    outlen = math.sqrt(outx * outx + outy * outy)
    return (1.0 * inx * outx + 1.0 * iny * outy) / (inlen * outlen)

    
def side(invec, outvec):
    """
    Desc: 计算outvec相对于invec的位置, 左侧left, 右侧right, 如果是中间, 反悔
    Args: 
        invec: invec, (inx, iny)
        outvec: outvec, (outx, outy)
    Returns:
        1: left, -1: right, 0: 同向, -2, 逆向
    Raises: 
        error.RegionError
    """
    (inx, iny) = invec
    (outx, outy) = outvec
    if (inx == 0 and iny == 0) or (outx == 0 and outy == 0):
        raise error.RegionError("vector is zero")
    side1 = inx * outy - outx * iny
    cos1 = inx * outx + iny * outy
    if side1 < 0:    # outlink is right of inlink
        return -1
    elif side1 > 0:  # outlink is left of inlink
        return 1
    elif cos1 > 0:   # side == 0 outlink is the same direction
        return 1
    else:    # side = 0, outlink is inlink's anti extention
        return -1


class Node(mygeo.Point):
    """
    Desc: 区域生成的Point对象
    """
    def __init__(self, x, y):
        """
        Desc: 节点对象
        Args:
            self : self
            x : x
            y : y
        Return: 
            None 
        Raise: 
            None
        """
        mygeo.Point.__init__(self, x, y)
        self.in_links = []
        self.out_links = []


class Link(mygeo.Segment):
    """
    Desc: 区域生成的线段对象
    """
    def __init__(self, start, end):
        """
        Desc: 用于面积生成split_seg
        Args:
            self : self
            start : Node 
            end : Node
        Return: 
            None 
        Raise: 
            None
        """
        mygeo.Segment.__init__(self, start, end)
        start.out_links.append(self)
        end.in_links.append(self)
        self.used = False
        
    def leftest(self):
        """
        Desc: 获取最左侧link
        Args:
            self : self
        Return: 
            Link object
        Raise: 
            None
        """
        min_cs = 5.0
        min_link = None
        for link in self.end.out_links:
            out_vec = link.vec()
            in_vec = self.vec()
            side1 = side(in_vec, out_vec)
            cos1 = cos(in_vec, out_vec)
            cs = side1 * cos1 + (1 - side1)
            if cs < min_cs:
                min_cs = cs
                min_link = link
        if min_link is None:
            raise error.RegionError("there is not a leftest link")
        return min_link
    
    def vec(self):
        """
        Desc: seg构建的向量, end - start构成了这个seg的向量
        Args:
            self : self
        Return: 
            vec, tuple of (x, y)
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object has not vec")
        return (self.end.x - self.start.x, self.end.y - self.start.y)

    def reverse(self):
        """
        Desc: link反转, 获取当前link的一个反link, 即start, end对调
        Args:
            self : self
        Return: 
            a new link
        Raise: 
            None
        """
        return Link(self.end, self.start)


class ValueDict(object):
    """
    Desc: value dict, 查询的元素不存在, 就会给这个dict添加这个元素
    """
    def __init__(self):
        """
        Desc: 初始化
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.value_dict = {}

    def find(self, key, value=None):
        """
        Desc: 添加一个key, 如果key已经存在, 返回key对应的value
        Args:
            key : key
            value : value, 如果value == None, 则认为value = key
        Return: 
            key对应的value
        Raise: 
            None
        """
        key_str = key
        if key_str in self.value_dict:
            return self.value_dict[key_str]
        else:
            if value is None:
                val = key
            else:
                val = value
            self.value_dict[key_str] = val
            return val

    def is_in(self, key, value=None):
        """
        Desc: 添加一个key, 如果key已经存在, 返回key对应的value
        Args:
            key : key
            value : value, 如果value == None, 则认为value = key
        Return: 
            True, if is in
            False, if not in, and add key, value
        Raise: 
            None
        """
        key_str = key
        if key_str in self.value_dict:
            return True
        else:
            if value is None:
                val = key
            else:
                val = value
            self.value_dict[key_str] = val
            return False
        

class RegionGenerator(object):
    """
    Desc: region generator
    """
    def __init__(self, segs):
        """
        Desc: 初始化
        Args:
            self : self
            segs : seg之间无重复, 除seg顶点外, 无其它的交点
        Return: 
            None 
        Raise: 
            None
        """
        self.links = []
        self.node_dict = ValueDict()
        self.link_dict = {}
        for seg in segs:
            start = self.node_dict.find(seg.start.askey(), Node(seg.start.x, seg.start.y))
            end = self.node_dict.find(seg.end.askey(), Node(seg.end.x, seg.end.y))
            if start == end:     # seg.start seg.end 非常接近的时候
                continue
            link = self.__get_link(start, end)
            if link is not None:
                self.links.append(link)
                self.links.append(link.reverse())

    def __get_link(self, start, end):
        """
        Desc: 根据start, end node生成一个link信息
        Args:
            self : self
            start : Node
            end : Node
        Return: 
            Link, 
        Raise: 
            None
        """
        ret = None
        segstr = mygeo.Segment(start, end).askey()
        if segstr not in self.link_dict:
            self.link_dict[segstr] = 1
            ret = Link(start, end)
        return ret
        
    def run(self):
        """
        Desc: 区域生成, 对于一个link经过同一个节点, 则形成环
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        ret = []
        for link in self.links:
            if not link.used:
                reg = self.get_a_region(link) 
                if reg is not None:
                    ret.append(reg)
        return ret
    
    def get_a_region(self, link):
        """
        Desc: 从一个link出发, 生成一个region
        Args:
            self : self
            link : 从这个link出发, 形成的一个region
        Return: 
            mygeo.Region
        Raise: 
            None
        """
        nodes = [link.start]
        next = link
        while not next.used:
            nodes.append(next.end)
            next.used = True
            next = next.leftest()
        if not nodes[0] == nodes[-1]:
            raise error.RegionError("first node is not equal to end node")
        points = None
        holes = []
        node_dict = {}
        temp_holes = []
        i = 0
        while i < len(nodes):
            nd = nodes[i]
            ndstr = nd.askey()
            if ndstr in node_dict:
                start_idx = node_dict[ndstr]
                sub_nodes = nodes[start_idx: i]
                if len(sub_nodes) >= 3:
                    region = mygeo.Region(sub_nodes)
                    if region.area > 0:
                        if mygeo.is_counter_clockwise(sub_nodes):
                            if points is None:
                                points = sub_nodes
                            else:
                                raise error.RegionError("too many ring is counter clockwise")
                        else:
                            holes.append(sub_nodes)
                    else:
                        temp_holes.append(sub_nodes)
                nodes = nodes[:start_idx] + nodes[i:]
                i = start_idx
            else:
                node_dict[ndstr] = i
            i += 1
        if points is not None:
            return mygeo.Region(points, holes)
        else:
            return None

                    
def segments_to_points(segments):
    """
    Desc: 从segments中提取point, 放到points集合中
    Args:
        segments : list of mygeo.Segment
    Return: 
        list of mygeo.Point
    Raise: 
        None
    """
    ret = []
    point_dict = ValueDict()
    for seg in segments:
        if not point_dict.is_in(seg.start.askey(), seg.start):
            ret.append(seg.start)
        if not point_dict.is_in(seg.end.askey(), seg.end):
            ret.append(seg.end)
    return ret


def segments_to_cluster_points(segments):
    """
    Desc: 从segments中提取point, 放到points集合中
    Args:
        segments : list of mygeo.Segment
    Return: 
        list of cluster.Point
    Raise: 
        None
    """
    ret = []
    point_dict = ValueDict()
    for seg in segments:
        if not point_dict.is_in(seg.start.askey(), seg.start):
            ret.append(cluster.Point(seg.start.x, seg.start.y))
        if not point_dict.is_in(seg.end.askey(), seg.end):
            ret.append(cluster.Point(seg.end.x, seg.end.y))
    return ret


def regions_to_points(regions):
    """
    Desc: 根据region生成point
    Args:
        regions : regions
    Return: 
        None 
    Raise: 
        None
    """
    ret = []
    point_dict = ValueDict()
    for region in regions:
        for pt in region.points:
            if not point_dict.is_in(pt.askey(), pt):
                ret.append(pt)
        for hole in region.holes:
            for pt in hole:
                if not point_dict.is_in(pt.askey(), pt):
                    ret.append(pt)
    return ret


def regions_to_cluster_points(regions):
    """
    Desc: 根据region生成point
    Args:
        regions : regions
    Return: 
        list of cluster.Point
    Raise: 
        None
    """
    ret = []
    id = 0
    point_dict = ValueDict()
    for region in regions:
        id += 1
        for pt in region.points:
            if not point_dict.is_in(pt.askey(), pt):
                ret.append(cluster.Point(pt.x, pt.y, id))
        for hole in region.holes:
            for pt in hole:
                if not point_dict.is_in(pt.askey(), pt):
                    ret.append(cluster.Point(pt.x, pt.y, id))
    return ret


def clusters_to_pointmap(clusters):
    """
    Desc: 根据cluster构建pointmap, ValueDict对象
    Args:
        clusters : clusters
    Return: 
        ValueDict
    Raise: 
        None
    """
    point_dict = ValueDict()
    for cluster in clusters:
        center = cluster.center()
        for ptstr in cluster.points:
            point_dict.is_in(ptstr, center)
    return point_dict


def __segment_2_newseg(segment, point_dict, seg_dict):
    """
    Desc: 根据segment计算修正后的segment, 然后根据seg_dict进行去重
    Args:
        segment : segment
        point_dict : point dict
        seg_dict : seg dict
    Return: 
        a new segment
    Raise: 
        None
    """
    start = point_dict.find(segment.start.askey(), segment.start)
    end = point_dict.find(segment.end.askey(), segment.end)
    start.trunc()
    end.trunc()
    if start == end:
        return None
    newseg = mygeo.Segment(start, end)
    if seg_dict.is_in(newseg.askey(), newseg):
        return None
    return newseg


def simplify_by_pointmap(segments, regions, point_dict):
    """
    Desc: 从point_dict中获取seg修正后的起终点, 然后返回新的segment
    Args:
        segments : segments
        regions : regions
        pointmap : pointmap
    Return: 
        segs, 所有合法的seg, 去重
    Raise: 
        None
    """
    ret = []
    seg_dict = ValueDict()
    for seg in segments:
        newseg = __segment_2_newseg(seg, point_dict, seg_dict)
        if newseg is not None:
            ret.append(newseg)
    if regions is not None:
        for region in regions:
            for seg in region.segments():
                newseg = __segment_2_newseg(seg, point_dict, seg_dict)
                if newseg is not None:
                    ret.append(newseg)
    return ret


class RegionAttract(object):
    """
    Desc: 区域向周边seg进行吸附
    """
    def __init__(self, regions, width):
        """
        Desc: 初始化
        Args:
            self : self
            regions : 需要做吸附的region list
            width : 吸附的宽度阈值, 必须是int类型
        Return: 
            None 
        Raise: 
            None
        """
        grid_dict = {}
        points = regions_to_points(regions)
        point_map = ValueDict()
        for pt in points:
            grid = pt.grid(width)
            item = [pt, None, width]
            if point_map.is_in(pt.askey(), item):
                continue
            for grid_x in range(grid[0] - 1, grid[0] + 2):
                for grid_y in range(grid[1] - 1, grid[1] + 2):
                    key_grid = (grid_x, grid_y)
                    if key_grid in grid_dict:
                        grid_dict[key_grid].append(item)
                    else:
                        grid_dict[key_grid] = [item]
        self.grid_dict = grid_dict
        self.regions = regions
        self.point_map = point_map
        self.width = width

    def __modify_ring(self, points):
        """
        Desc: 修正一个环上的所有点
        Args:
            self : self
            points : 需要修正的点
            point_map : point_map
        Return: 
            list of mygeo.Point
        Raise: 
            None
        """
        new_points = []
        for pt in points:
            item = self.point_map.find(pt.askey(), None)
            if item is None or item[1] is None:
                new_points.append(pt)
            else:
                new_points.append(item[1])
        return new_points
    
    def __project(self, seg, point):
        """
        Desc: 计算点在seg上的投影点, 以及点到seg投影点的距离
        Args:
            seg : mygeo.Segment
            point : mygeo.Point
        Return: 
            (project_point, dist)
        Raise: 
            None
        """
        l1 = seg.linestring()
        d1 = l1.project(point.point())
        pt = l1.interpolate(d1)
        return mygeo.Point(pt.x, pt.y)

    def run(self, segs):
        """
        Desc: 将现有的regions向segs进行吸附
        Args:
            self : self
            segs : segs
        Return: 
            list of new regions
        Raise: 
            None
        """
        # 计算点在seg上的投影点, 和点到投影点的距离
        for seg in segs:
            for grid in seg.grids(self.width):
                if grid not in self.grid_dict:
                    continue
                for item in self.grid_dict[grid]:
                    (pt, spt, min_dist) = item
                    dist = mygeo.pt_2_seg_dist(pt, seg)
                    if dist < min_dist:
                        item[1] = self.__project(seg, pt)
                        item[2] = dist
        # 基于point_map构建
        ret = []
        for reg in self.regions:
            points = self.__modify_ring(reg.points)
            holes = []
            for hole in reg.holes:
                holes.append(self.__modify_ring(hole))
            newreg = None
            try:
                newreg = mygeo.Region(points, holes)
            except error.RegionError:
                error.debug("region not valid: %s" % (str(reg)))
            if newreg is not None and newreg.polygon().is_valid:
                ret.append(newreg)
            else:
                ret.append(reg)
        return ret

