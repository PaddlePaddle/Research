# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 

import sys
import threading

from region import geometry as mygeo
from region import error

class Link(mygeo.Segment):
    """
    Desc: 区域处理算法的link对象
    """
    __slots__ = ("twin_link", "region")

    def __init__(self, start, end):
        """
        Desc: link就是一个mygeo.Segment, 记录了该link对应的region和twinlink
              region在link的左侧, twinlink是该link的反向link, 
              twinlink对应的region与该link对应的region相邻
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        mygeo.Segment.__init__(self, start, end)
        self.twin_link = None
        self.region = None
    
    def twin_link_region(self):
        """
        Desc: 获取该link的twin region, twin link对应的region
        Args:
            self : self
        Return: 
            Region, or None
        Raise: 
            None
        """
        ret = None
        if self.twin_link is not None:
            ret = self.twin_link.region
        return ret


class Neighbor(object):
    """
    Desc: region的邻居对象, 记录邻居的region, 同邻居的边界信息等
    """
    __slots__ = ("region", "points", "length")

    def __init__(self, region, points, length):
        """
        Desc: neighbor的三元组
        Args:
            self : self
            region : region, Region
            points : touch的points, list of mygeo.Point, 同邻居的边界信息
            length : touch部分的长度
        Return: 
            None 
        Raise: 
            None
        """
        self.region = region
        self.points = points
        self.length = length

    def merge(self, points, length):
        """
        Desc: 将一段points, 增加到已有的neighbor对象中
              条件 points[-1] == self.points[0]
        Args:
            self : self
            points : 需要添加的points, list of mygeo.Point
            length : points的长度
        Return: 
            None 
        Raise: 
            None
        """
        if not points[-1] == self.points[0]:
            raise error.RegionError("can't merge neighbor")
        newpoints = []
        newpoints.extend(points)
        newpoints.extend(self.points[1:])
        self.points = newpoints
        self.length += length

    def display(self):
        """
        Desc: 打印neighbor对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        print "neighbor"
        if self.region is not None:
            print "\tregion: ", str(self.region)
        else:
            print "\tregion: None"
        print "\tPoint: ",
        for pt in self.points:
            print str(pt), ", ",
        print
        print "\tlength: ", self.length


class LinkDict(object):
    """
    Desc: link生成装置
    """
    __slots__ = ("link_dict", )

    def __init__(self):
        """
        Desc: 初始化, link集合为空
        Args:
            self :
        Return: 
            None 
        Raise: 
            None
        """
        self.link_dict = {}

    def clear(self):
        """
        Desc: 清空link dict
        
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.link_dict = {}

    def get_link(self, start, end):
        """
        Desc: 从link_dict中获取(start, end)节点构建的link
        Args:
            self : self
            start : 起点, mygeo.Point
            end : 终点, mygeo.Point
        Return: 
            start, end节点组成的link
        Raise: 
            None
        """
        ret = None
        link = Link(start, end)
        linkstr = link.askey()
        if linkstr in self.link_dict:
            ret = self.link_dict[linkstr]
            if not start == ret.start:
                ret = ret.twin_link
        else:
            twin_link = Link(end, start)
            link.twin_link = twin_link
            twin_link.twin_link = link
            self.link_dict[linkstr] = link
            ret = link
        return ret

    def del_link(self, start, end):
        """
        Desc: 从link dict中删除一个link
        Args:
            self : self
            start : start
            end : end
        Return: 
            None 
        Raise: 
            None
        """
        linkstr = Link(start, end).askey()
        if linkstr in self.link_dict:
            del self.link_dict[linkstr]

    def segments(self):
        """
        Desc: 获取所有的segment
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        ret = []
        for link in self.link_dict.values():
            ret.append(mygeo.Segment(link.start, link.end))
        return ret


def points_2_segs(points):
    """
    Desc: 点转换为seg
    Args:
        points : points
    Return: 
        list of mygeo.Segment
    Raise: 
        None
    """
    ret = []
    lastpt = points[0]
    for pt in points[1:]:
        ret.append(mygeo.Segment(lastpt, pt))
        lastpt = pt
    return ret


class Region(mygeo.Region):
    """
    Desc: 记录region与link之间的关系
    """
    __slots__ = ("link_dict", "parent_region", "linkss", "area_", "length_", "width_", \
                 "neighbors")

    def __init__(self, region, parent_region, link_dict):
        """
        Desc: 初始化
        Args:
            self : self
            region : region, mygeo.Region
            link_dict : link生成装置
            raw_region : 如果该region被包含在一个原始region中
            links : region对应的links
        Return: 
            None 
        Raise: 
            None
        """
        mygeo.Region.__init__(self, region.points, region.holes)
        self.link_dict = link_dict
        self.parent_region = parent_region
        self.linkss = self.__get_linkss()
        self.area_ = region.area()
        self.length_ = region.length()
        self.width_ = self.area_ / self.length_
        self.neighbors = None

    def __trans_points_to_links(self, points):
        """
        Desc: 将一组points list转化成link list
        Args:
            link_dict : 所有link的词典
            points : points list
        Return: 
            list of link 
        Raise: 
            None
        """
        ret = []
        lastpt = points[-1]
        for pt in points:
            link = self.link_dict.get_link(lastpt, pt)
            link.region = self
            ret.append(link)
            lastpt = pt
        return ret

    def __get_linkss(self):
        """
        Desc: 将region分解成link list
        Args:
            regions : list of region
        Return: 
            list of list of Link, links is all link in a region
        Raise: 
            None
        """
        links = self.__trans_points_to_links(self.points)
        ret = [links]
        for hole in self.holes:
            links = self.__trans_points_to_links(hole)
            ret.append(links)
        return ret
        
    def __compute_ring_neighbors(self, links):
        """
        Desc: 只针对一个ring构建的所有links, 计算这个ring对应的neighbors
        Args:
            self : self
            links : 一个ring构建的所有links
        Return: 
            list of Neighbor(region, points, length), 
                region 接触的region, 
                points 接触的points, 
                length 接触部分的长度
        Raise: 
            None
        """
        ret = []
        if len(links) == 0:
            return ret
        points = []
        length = 0
        last_region = -1
        for link in links:
            linklen = link.length()
            region = link.twin_link_region()
            if region == last_region:
                points.append(link.end)
                length += linklen
            else:
                if last_region != -1:
                    ret.append(Neighbor(last_region, points, length))
                points = [link.start, link.end]
                length = linklen
                last_region = region
        if last_region != -1:
            if len(ret) > 1 and last_region == ret[0].region:
                ret[0].merge(points, length)
            else:
                ret.append(Neighbor(last_region, points, length))
        return ret
        
    def refresh_neighbors(self):
        """
        Desc: 计算该region接触的所有region
        Args:
            self : self
        Return: 
            list of Neighbor(region, points, length), 
                region 接触的region, 
                points 接触的points, 
                length 接触部分的长度
        Raise: 
            None
        """
        ret = []
        for links in self.linkss:
            neighbors = self.__compute_ring_neighbors(links)
            ret.extend(neighbors)
        self.neighbors = ret

    def is_small_region(self, area_thres, width_thres):
        """
        Desc: 判断当前region是否是一个小region
        Args:
            self : self
            area_thers : 面积阈值
            width_thers : 宽度阈值
        Return: 
            None 
        Raise: 
            None
        """
        return self.area_ <= area_thres or self.width_ <= width_thres
            
    def merge_small_region(self, area_thres, width_thres):
        """
        Desc: 获取当前region中需要删除的seg
        Args:
            self : self
            area_thers : 面积阈值
            width_thers : 宽度阈值
        Return: 
            list of mygeo.Segment
        Raise: 
            None
        """
        small = True   # 当前是small region
        max_length = 0
        max_region = None
        for neighbor in self.neighbors:
            if neighbor.region is None:
                continue
            if not neighbor.region.is_small_region(area_thres, width_thres):
                if small or neighbor.length > max_length:
                    small = False
                    max_length = neighbor.length
                    max_region = neighbor.region
            elif small and neighbor.length > max_length:
                max_length = neighbor.length
                max_region = neighbor.region
        if max_region is not None:
            for nb in self.neighbors:
                if nb.region == max_region:
                    return points_2_segs(nb.points)
        return None
    
    def merge_sibling_region(self):
        """
        Desc: 获取一个兄弟区域, two regions that have the same parent are siblings
        Args:
            self : self
        Return: 
            list of mygeo.Segment
        Raise: 
            None
        """
        if self.parent_region is None:
            return None
        ret = []
        for nb in self.neighbors:
            if nb.region is not None and nb.region.parent_region == self.parent_region:
                ret.extend(points_2_segs(nb.points))
        return ret

    def __replace_ring(self, old_points, src_points, dest_points):
        """
        Desc: 替换一个ring中的points到newpoints
        Args:
            self : self
            old_points : ring的所有points
            src_points : old points
            dest_points : new points
        Return: 
            new_points : 替换后的points列表
        Raise: 
            None
        """
        new_points = old_points
        old_length = len(old_points)
        length = len(src_points)
        for i in range(0, old_length):
            found = True
            for j in range(0, length):
                if not src_points[j] == old_points[(i + j) % old_length]:
                    found = False
                    break
            if found:
                if length > old_length:
                    new_points = dest_points[: -1]
                elif (i + length) >= old_length:
                    new_points = old_points[(i + length) % old_length: i] + dest_points
                else:
                    new_points = old_points[:i] + dest_points + old_points[i + length:]
                break
        return new_points

    def replace(self, src_points, dest_points):
        """
        Desc: 顶点替换, 将区域中的points点序列替换成newpoints点序列
        Args:
            self : self
            src_points : old points 
            dest_newpoints : new points
        Return: 
            a new region, mygeo.Region
            如果没有找到points, 则会返回同原region一样的region
        Raise: 
            None
        """
        newpoints = self.__replace_ring(self.points, src_points, dest_points)
        newholes = []
        for hole in self.holes:
            newhole = self.__replace_ring(hole, src_points, dest_points)
            newholes.append(newhole)
        return mygeo.Region(newpoints, newholes)


def build_region_dict(regions, grid_size, key=None):
    """
    Desc: 构建region的网格化字典, 根据网格能够查询到该网格对应的region
    Args:
        regions : list of mygeo.Region
        grid_size : 网格的大小
        key : 生成region
    Return: 
        dict of grid -> region list
    Raise: 
        None
    """
    if key is None:
        key = lambda x: x
    grid_dict = {}
    for reg in regions:
        for grid in key(reg).grids(grid_size):
            if grid in grid_dict:
                grid_dict[grid].append(reg)
            else:
                grid_dict[grid] = [reg]
    return grid_dict


class RegionMerger(object):
    """
    Desc: 将小区域向大区域进行合并, 原始区域包含的一些小区域也可合并
    """
    def __init__(self, regions, raw_regions):
        """
        Desc: 初始化
        Args:
            self : self
            regions : 需要合并的区域
            raw_regions : 原始region, 这些region有可能被分裂了, 进行合并
        Return: 
            None 
        Raise: 
            None
        """
        self.regions = regions
        self.raw_regions = raw_regions
        self.raw_region_dict = None
        self.link_dict = LinkDict()

    def __get_raw_region(self, region, grid_size):
        """
        Desc: 计算当前region的grids, 
              每个grid计算当前region与grid内的raw_region进行计算
        Args:
            self : self
            region : 当前region
        Return: 
            该region隶属的raw_region
        Raise: 
            None
        """
        if self.raw_region_dict is None:
            return None
        center = region.center()
        grid = center.grid(grid_size)
        if grid in self.raw_region_dict:
            raw_regions = self.raw_region_dict[grid]
            for rawreg, mbr in raw_regions:
                if mbr[1].x >= center.x and mbr[1].y >= center.y \
                   and mbr[0].x <= center.x and mbr[0].y <= center.y:
                    if rawreg.polygon().covers(center.point()):
                        return rawreg
        return None

    def __trans_regions(self, grid_size):
        """
        Desc: 计算转换后regions, 将输入region转换为新的Region
        Args:
            self : self
            grid_size : 网格大小, 用于判断region是否属于某个raw region
        Return: 
            转换后的region list
        Raise: 
            None
        """
        ret = []
        if self.raw_regions is not None:
            raw_regions = [(r, r.mbr()) for r in self.raw_regions]
            self.raw_region_dict = build_region_dict(raw_regions, grid_size, lambda x:x[0])
        for region in self.regions:
            raw_region = self.__get_raw_region(region, grid_size)
            newreg = Region(region, raw_region, self.link_dict)
            ret.append(newreg)
        for reg in ret:
            reg.refresh_neighbors()
        return ret

    def run(self, grid_size, area_thres, width_thres):
        """
        Desc: 根据给定的阈值, 进行区域合并
        Args:
            grid_size : 网格大小, 用于判断两个区域之间的包含, 相交关系
            area_thres : 面积阈值, 低于这个面积的区域会被合并
            width_thres : area / length阈值, 低于此阈值的区域会被合并
        Return: 
            list of mygeo.Region
        Raise: 
            None
        """
        regions = self.__trans_regions(grid_size) # mygeo.Region -> Region
        for region in regions:
            segs = region.merge_sibling_region()
            if segs is not None:
                for seg in segs:
                    self.link_dict.del_link(seg.start, seg.end)
            elif region.is_small_region(area_thres, width_thres):
                segs = region.merge_small_region(area_thres, width_thres)
                if segs is not None:
                    for seg in segs:
                        self.link_dict.del_link(seg.start, seg.end)
        return self.link_dict.segments()


class RegionFilter(object):
    """
    Desc: 将包含的region删除, 将有重叠的region进行修正, 保证不重叠
    """
    def __init__(self, regions):
        """
        Desc: 初始化
        Args:
            self : self
            regions : 区域
        Return: 
            None 
        Raise: 
            None
        """
        self.regions = regions

    def __delete_contain(self, grid_dict):
        """
        Desc: 删除被其它区域包含的区域, 如果一个区域一半以上都在另一个区域中,则认为被包含
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        total = len(grid_dict)
        for dict_regions in grid_dict.values():
            regions = [r for r in dict_regions if not r[0].is_empty()]
            regions.sort(key=lambda x: x[1])
            for i in range(len(regions) - 1):
                child_reg, child_area, child_mbr = regions[i]
                for (parent_reg, parent_area, parent_mbr) in regions[i + 1:]:
                    if parent_mbr[1].x >= child_mbr[1].x and parent_mbr[1].y >= child_mbr[1].y \
                      and parent_mbr[0].x <= child_mbr[0].x and parent_mbr[0].y <= child_mbr[0].y:
                        try:
                            if (parent_reg.polygon().covers(child_reg.polygon())):
                                child_reg.destroy()
                        except error.RegionError:
                            continue
            
    def run(self, grid_size):
        """
        Desc: 运行region filter
        Args:
            self : self
        Return: 
            region list after filter
        Raise: 
            None
        """
        grid_dict = {}
        for reg in self.regions:
            area = reg.area()
            mbr = reg.mbr()
            for grid in reg.grids(grid_size):
                if grid in grid_dict:
                    grid_dict[grid].append((reg, area, mbr))
                else:
                    grid_dict[grid] = [(reg, area, mbr)]
        self.__delete_contain(grid_dict)
        return [r for r in self.regions if not r.is_empty()]

    def __thread_start(self, func, grid_dict_list):
        """
        Desc: 启动func的多线程函数
        Args:
            self : self
            func : 一个线程的启动函数
            grid_dict_list : 所有grid_dict
        Return: 
            None 
        Raise: 
            None
        """
        thds = []
        for gd in grid_dict_list:
            t = threading.Thread(target=func, args=(gd,))
            t.start()
            thds.append(t)
        for t in thds:
            t.join()

    def multi_run(self, grid_size, thread_num=4):
        """
        Desc: 多线程运行区域过滤
        Args:
            self : self
            grid_size : grid_size
        Return: 
            None 
        Raise: 
            None
        """
        grid_dict_list = []
        for i in range(thread_num):
            grid_dict_list.append({})
        for reg in self.regions:
            area = reg.area()
            mbr = reg.mbr()
            for grid in reg.grids(grid_size):
                idx = (grid[0] + grid[1]) % thread_num
                grid_dict = grid_dict_list[idx]
                if grid in grid_dict:
                    grid_dict[grid].append((reg, area, mbr))
                else:
                    grid_dict[grid] = [(reg, area, mbr)]
        self.__thread_start(self.__delete_contain, grid_dict_list)
        return [r for r in self.regions if not r.is_empty()]


class RegionFinder(object):
    """
    Desc: 根据一组regions, 寻找点所在的region

          regions = []

          regions.append((id, Region(...))

          rf = RegionFinder(regions, 1024)

          id = rf.find(Point(2.34, 4.56))

    """
    def __init__(self, regions, grid_size):
        """
        Desc: 初始化

        Args:
            self : self

            regions : list of (id, region)

            grid_size : 网格大小
        Return: 
            None 
        Raises: 
            None
        """
        self.grid_size = grid_size
        self.region_dict = build_region_dict(regions, grid_size, key=lambda x: x[1])

    def find_region(self, point):
        """
        Desc: 获取当点的region id

        Args:
            self : self

            point : 当前点
        Return: 
            region id
        Raises: 
            None
        """
        grid = point.grid(self.grid_size)
        if grid not in self.region_dict:
            return None
        for id, region in self.region_dict[grid]:
            if region.polygon().covers(point.point()):
                return id
        return None


