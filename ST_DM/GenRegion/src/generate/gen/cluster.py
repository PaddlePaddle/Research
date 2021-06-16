# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys

from region import error
from region import geometry as mygeo

class Point(mygeo.Point):
    """
    Desc: 聚类的point对象，id, x, y; id相同的点不能聚合在一起
    """
    __slots__ = ("id", )

    def __init__(self, x, y, id=None):
        mygeo.Point.__init__(self, x, y)
        self.id = id

class Cluster(object):
    """
    Desc: 聚类对象, 点聚类后的结果
    """
    __slots__ = ("minx", "maxx", "miny", "maxy", "points", "ids")

    def __init__(self):
        """
        Desc: 一个聚类的结果对象, 一个聚类的结果包括: 1) MBR, 2) points
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.minx = None
        self.maxx = None
        self.miny = None
        self.maxy = None
        self.points = set([])
        self.ids = set([])

    def count(self):
        """
        Desc: 统计元素的数量
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        return len(self.points)

    def destroy(self):
        """
        Desc: 销毁这个cluster对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.points.clear()
        self.ids.clear()
        self.minx = self.maxx = self.miny = self.maxy = None

    def is_empty(self):
        """
        Desc: 判断这个对象是否为空对象
        Args:
            self : self
            point : Point
        Return: 
            True, if empty; False, if not empty
        Raise: 
            None
        """
        return self.minx is None

    def add_point(self, point, is_vip=True):
        """
        Desc: 向聚类中添加新的点
        Args:
            self : self
            point : 新增的点, Point
            is_vip : 是否是vip点, vip点会影响center的计算
            id : point id, 相同id的point不能聚合在一起
        Return: 
            None 
        Raise: 
            None
        """
        self.points.add(point.askey())
        if is_vip:
            if self.minx is None or point.x < self.minx:
                self.minx = point.x
            if self.maxx is None or point.x > self.maxx:
                self.maxx = point.x
            if self.miny is None or point.y < self.miny:
                self.miny = point.y
            if self.maxy is None or point.y > self.maxy:
                self.maxy = point.y
        if point.id is not None:
            self.ids.add(point.id)

    def merge(self, cl):
        """
        Desc: 两个cluster进行合并
        Args:
            self : self
            cl : new cluster
        Return: 
            None 
        Raise: 
            None
        """
        if cl.is_empty():
            return
        if not self.can_merge(cl):
            raise error.RegionError("cluster id conflict")
        if self.is_empty():
            self.minx, self.maxx, self.miny, self.maxy = cl.minx, cl.maxx, cl.miny, cl.maxy
        else:
            self.minx = min(self.minx, cl.minx)
            self.maxx = max(self.maxx, cl.maxx)
            self.miny = min(self.miny, cl.miny)
            self.maxy = max(self.maxy, cl.maxy)
        for pt in cl.points:
            self.points.add(pt)
        for id in cl.ids:
            self.ids.add(id)

    def can_merge(self, cl):
        """
        Desc: 是否能够merge
        Args:
            self : self
            cl : cl
        Return: 
            None 
        Raise: 
            None
        """
        for id in cl.ids:
            if id in self.ids:
                return False
        return True

    def in_cluster(self, point):
        """
        Desc: 判断点是否在cluster中 
        
        Args:
            self : self
            point : Point
        Return: 
            True, if point in cluster, otherwise False
        Raise: 
            None
        """
        return (point.askey() in self.points)

    def center(self):
        """
        Desc: 获取聚类的中心点, 聚类的中心点是类中所有vip点mbr的中心点
        Args:
            self : self
        Return: 
            mygeo.Point(x, y) 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty cluster hasn't center()")
        return mygeo.Point((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

    def width_x(self):
        """
        Desc: 计算这个聚类的x宽度
        Args:
            self : self
        Return: 
            float, x方向上的宽度
            如果类为空, 返回None
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty cluster hasn't width_x")
        return self.maxx - self.minx

    def width_y(self):
        """
        Desc: 计算这个聚类的y宽度
        Args:
            self : self
        Return: 
            float, y方向上的宽度
            如果类为空, 返回None
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty cluster hasn't width_y")
        return self.maxy - self.miny


def is_friend_dist(dist, width):
    """
    Desc: 判断dist是否符合friend的条件
    Args:
        self : self
        dist : (dist_x, dist_y)
    Return: 
        True, if friend dist, False, if not friend dist
    Raise: 
        None
    """
    return dist[0] <= width and dist[1] <= width


class HCCluster(Cluster):
    """
    Desc: 基于层次聚类的cluster对象
    """
    def __init__(self):
        """
        Desc: 初始化, 层次聚类中增加了friend的概念, 每个类周边的类为它的friend
              friend的的定义: 如果两个类A, B合并后的类大小没有超过阈值, 则这两个类为friend
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        Cluster.__init__(self)
        # (cluster, dist)
        # best friend is self.friends[0]
        self.friends = []

    def collect_best_friend(self):
        """
        Desc: 从已有的friends中选择best friend放在self.friends的首部
              best friend可能不止一个
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        min_weight = sys.float_info.max
        for friend in self.friends:
            weight = friend[1]
            if weight < min_weight:
                min_weight = weight
        swap_count = 0
        i = 0
        while i < len(self.friends):
            if self.friends[i][1] == min_weight:
                self.friends[swap_count], self.friends[i] = self.friends[i], self.friends[swap_count]
                swap_count += 1
            i += 1

    def del_friends(self, del_frds):
        """
        Desc: 从friends列表中删除部分del_frds, 如果删除了首元素, 则需要重新选举首元素
        Args:
            self : self
            del_frds : 需要删除的friends列表
        Return: 
            True, if best friend changed, otherwise False
        Raise: 
            None
        """
        i = 0
        first_del = False
        while i < len(self.friends): 
            hc, weight = self.friends[i]
            if hc in del_frds:
                self.friends.pop(i)
                i -= 1
                if i == 0:
                    first_del = True
            i += 1
        if first_del and len(self.friends) > 1:
            self.collect_best_friend()
        return first_del

    def bfs_merge(self, width):
        """
        Desc: BFS merge, 判断自己的best friend是否也是自己, 如果是就merge
        Args:
            self : self
            width : 聚类阈值
        Return: 
            发生了merge, return True
            没有发生merge, return False
        Raise: 
            None
        """
        best_frds = self.best_friends()
        for hc, weight in best_frds:
            frds = hc.best_friends()
            for (hc2, weight) in frds:
                if self == hc2:
                    self.merge(hc, width)
                    return True
        return False

    def merge(self, hc, width):
        """
        Desc: 需要合并的类
        Args: 
            self : self
            hc1 : 需要合并的类
        Return: 
            None 
        Raise: 
            None
        """
        # point合并
        Cluster.merge(self, hc)
        # friends合并
        new_friends = set([])
        for fhc, weight in self.friends:
            new_friends.add(fhc)
        for fhc, weight in hc.friends:
            new_friends.add(fhc)
        hc.destroy()
        self.friends = []
        for fhc in new_friends:
            if fhc == self or fhc == hc:
                continue
            fhc.del_friends([self, hc])
            dist = self.distance(fhc, width)
            if is_friend_dist(dist, width) and self.can_merge(fhc):
                self.add_friend(fhc, dist)
                fhc.add_friend(self, dist)

    def best_friends(self):
        """
        Desc: 寻找当前hc cluster的best friends, 返回所有best friends
        Args:
            self : self
        Return: 
            list of HCCluster
        Raise: 
            None
        """
        ret = []
        if len(self.friends) > 0:
            min_weight = self.friends[0][1]
            for friend in self.friends:
                if min_weight == friend[1]:
                    ret.append(friend)
        return ret

    def distance(self, hc, width):
        """
        Desc: 计算两个类之间的距离, 两个类距离的定义为: 
              cluster A 的宽度为(xa, ya), cluster B 的宽度为(xb, yb)
              A, B合并后的类cluster C 的宽度为(xc, yc), xc < width, yc < width
              if xa + ya > xb + yb: 
                  delta_x = xc - xa
                  delta_y = yc - ya 
              else
                  delta_x = xc - xb
                  delta_y = yc - yb 
        Args:
            self : self
            hc : the other hc cluster
            width : threshold
        Return: 
            (dist_x, dist_y) 
        Raise: 
            None
        """
        if hc.is_empty() or self.is_empty():
            raise error.RegionError("empty object hasn't distance")
        minx = min(self.minx, hc.minx)
        maxx = max(self.maxx, hc.maxx)
        miny = min(self.miny, hc.miny)
        maxy = max(self.maxy, hc.maxy)
        width_x = maxx - minx
        width_y = maxy - miny
        if width_x <= width and width_y <= width:
            if self.width_x() + self.width_y() > hc.width_x() + hc.width_y():
                return (width_x - self.width_x(), width_y - self.width_y())
            else:
                return (width_x - hc.width_x(), width_y - hc.width_y()) 
        return (width + 1, width + 1)

    def add_friend(self, hc, dist):
        """
        Desc: 添加一个朋友到self中, 计算self与hc之间的距离, 修正首元素
        Args:
            self : self
            hc : new hc cluster
            dist : 距离
        Return: 
            None
        Raise: 
            None
        """
        (dist_x, dist_y) = dist
        weight = dist_x + dist_y
        if len(self.friends) > 0 and weight <= self.friends[0][1]:
            self.friends.insert(0, (hc, weight))
        else:
            self.friends.append((hc, weight))

    def display(self):
        """
        Desc: 展现类的基本信息
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        print "hc cluster: ",
        for pt in self.points:
            print pt, ",",
        print
        for fhc, weight in self.friends:
            print "friend weight: ", weight, " : ",
            for pt in fhc.points:
                print pt, ",",
            print
        print "mbr: (%f, %f), (%f, %f)" % (self.minx, self.miny, self.maxx, self.maxy)


class HCAlgorithm(object):
    """
    Desc: 层次聚类
    """
    def __init__(self, width):
        """
        Desc: 层次聚类算法的实现
        Args:
            self : self
            width : 聚类mbr的阈值, 必须为int类型
        Return: 
            None 
        Raise: 
            None
        """
        self.width = width
        # 聚类的中间结果
        self.hcclusters = []
        # dict of (grid_id, HCCluster list)
        self.grids = {}

    def run(self, points):
        """
        Desc: 层次聚类算法的实现
        Args:
            self : self
            points : 所有的点
        Return: 
            list of clusters
        Raise: 
            None
        """
        # 每个点构建一个HCCluster
        self.build_clusters(points)
        # 构建点与点之间的friend关系
        self.build_friends()
        # 深度优先merge
        merge_count = 1
        while merge_count > 0:
            merge_count = 0
            for hc in self.hcclusters:
                if not hc.is_empty():
                    if hc.bfs_merge(self.width):
                        merge_count += 1
        # 从中间结果中得出最终的聚类结果
        ret = []
        for hc in self.hcclusters:
            if not hc.is_empty():
                ret.append(hc)
        return ret
        
    def build_clusters(self, points):
        """
        Desc: 每个点构建一个hccluster, 并分配到相应的grid中
        Args:
            self : slef
            points : list of mygeo.Point
        Return: 
            None 
        Raise: 
            None
        """
        for pt in points:
            hc = HCCluster()
            hc.add_point(pt)
            self.hcclusters.append(hc)
            grid = pt.grid(self.width)    # grid is a tuple (grid_x, grid_y) 
            if grid in self.grids:
                self.grids[grid].append(hc)
            else:
                self.grids[grid] = [hc]
    
    def build_friends(self):
        """
        Desc: 建立hccluster之间的朋友关系, 并记录到hccluster中
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        for grid in self.grids:
            grid_x, grid_y = grid
            self.build_friend_one_grid(grid)
            self.build_friend_two_grid(grid, (grid_x - 1, grid_y + 1))
            self.build_friend_two_grid(grid, (grid_x + 1, grid_y))
            self.build_friend_two_grid(grid, (grid_x, grid_y + 1))
            self.build_friend_two_grid(grid, (grid_x + 1, grid_y + 1))

    def build_friend_one_grid(self, grid):
        """
        Desc: 一个grid内部两两hccluster之间肯定是friend关系, 添加到friend关系中
        Args:
            self : self
            grid : 网格号(grid_x, grid_y)
        Return: 
            None 
        Raise: 
            None
        """
        hcs = self.grids[grid]
        count = len(hcs)
        for i in range(count - 1):
            hc1 = hcs[i]
            for j in range(i + 1, count):
                hc2 = hcs[j]
                dist = hc1.distance(hc2, self.width)
                if is_friend_dist(dist, self.width) and hc1.can_merge(hc2):
                    hc1.add_friend(hc2, dist)
                    hc2.add_friend(hc1, dist)

    def build_friend_two_grid(self, grid1, grid2):
        """
        Desc: 计算两个相邻grid之间的friend关系
        Args:
            self : self
            grid1 : 网格号(grid_x, grid_y)
            grid2 : 网格号(grid_x, grid_y)
        Return: 
            None 
        Raise: 
            None
        """
        if grid1 not in self.grids or grid2 not in self.grids:
            return
        hcs1 = self.grids[grid1]
        hcs2 = self.grids[grid2]
        for hc1 in hcs1:
            for hc2 in hcs2:
                dist = hc1.distance(hc2, self.width)
                if is_friend_dist(dist, self.width) and hc1.can_merge(hc2):
                    hc1.add_friend(hc2, dist)
                    hc2.add_friend(hc1, dist)
        

class Classifier(object):
    """
    Desc: 分类, 根据点与周边类的距离, 将点分到周边类中
    """
    def __init__(self, clusters, width):
        """
        Desc: 初始化
        Args:
            self : self
            clusters : clusters
            width : 类别的宽度, 类的宽度不能超过width
        Return: 
            None 
        Raise: 
            None
        """
        self.width = width
        self.grids = {}
        for cluster in clusters:
            center = cluster.center()
            lb = mygeo.Point(center.x - width / 2.0, center.y - width / 2.0)
            rt = mygeo.Point(center.x + width / 2.0, center.y + width / 2.0)
            lb_grid = lb.grid(width)
            rt_grid = rt.grid(width)
            for x in range(lb_grid[0], rt_grid[0] + 1):
                for y in range(lb_grid[1], rt_grid[1] + 1):
                    grid = (x, y)
                    if grid in self.grids:
                        self.grids[grid].append(cluster)
                    else:
                        self.grids[grid] = [cluster]
    
    def run(self, points):
        """
        Desc: 运行分类
        Args:
            self : self
            points : 需要分类的point
        Return: 
            list of mygeo.Point, 无法分类的point集合
        Raise: 
            None
        """
        left_points = []
        for pt in points:
            if not self.classify_point(pt):
                left_points.append(pt)
        return left_points

    def classify_point(self, point):
        """
        Desc: 将点分到相应的cluster中, 如果不能分到相应的cluster中, 则返回False
        Args:
            self : self
            point : 需要分类的mygeo.Point
        Return: 
            True, if point is classify into a cluster, otherwise False
        Raise: 
            None
        """
        grid = point.grid(self.width)
        ret = False
        if grid in self.grids:
            clusters = self.grids[grid]
            max_count = 0
            min_cluster = None
            for cluster in clusters:
                center = cluster.center()
                dist_x = abs(center.x - point.x)
                dist_y = abs(center.y - point.y)
                if dist_x <= self.width / 2.0 and dist_y <= self.width / 2.0 \
                        and point.id not in cluster.ids:
                    count = cluster.count()
                    if count > max_count:
                        min_cluster = cluster
                        max_count = count
            if min_cluster is not None:
                min_cluster.add_point(point, False)
                ret = True
        return ret


