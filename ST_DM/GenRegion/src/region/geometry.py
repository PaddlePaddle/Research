# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
import math

from shapely import wkt
from shapely import geometry as spgeo
from shapely import ops as spops

from region import error

class Point(object):
    """
    Desc: 点对象, 点坐标为浮点数
    """
    __slots__ = ("x", "y")

    # 坐标的精度, 如果是bd09mc坐标, 可以直接使用默认值
    # 如果是经纬度, 建议在用户侧代码的开始位置, 添加:
    # Point.precision = 5
    precision = 2
    base = math.pow(10, precision)
    
    @classmethod
    def set_precision(cls, precision):
        """
        Desc: 设置精度
        Args:
            precision : 精度, int
        Return: 
            None 
        Raise: 
            None
        """
        cls.precision = int(precision)
        cls.base = math.pow(10, cls.precision)

    def __init__(self, x, y):
        """
        Desc: 初始化
        Args:
            self : self对象
            x : float
            y : float
        Return: 
            None 
        Raise: 
            ValueError, if could not convert x,y to float
        """
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, pt):
        """
        Desc: 判断两个点是否相同, 两个点的x, y完全一致, 才认为两个点相同 
        Args:
            self : self
            pt : the other point
        Return: 
            True or False
        Raise: 
            None
        """
        return pt is not None and self.x == pt.x and self.y == pt.y

    def __str__(self):
        """
        Desc: Point对象的字符串表达
        Args:
            self : self对象
        Return: 
            point string
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't str()")
        format = "%%.%df %%.%df" % (Point.precision, Point.precision)
        return format % (self.x, self.y)

    def trunc(self):
        """
        Desc: 精度截断
        Args:
            self : self
        Return: 
            self 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't trunc")
        self.x = int(self.x * Point.base + 0.5) * 1.0 / Point.base
        self.y = int(self.y * Point.base + 0.5) * 1.0 / Point.base
        return self

    def askey(self):
        """
        Desc: Point作为dict的key的对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object hasn't key")
        return (self.x, self.y)

    def point(self):
        """
        Desc: 转换为spgeo.Point
        Args:
            self : self
        Return: 
            spgeo.Point
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't trans to spgeo.Point")
        return spgeo.Point(self.x, self.y)

    def grid(self, grid_size=1024):
        """
        Desc: 返回该对象的网格编号
              网格编号为字符串, 编号的格式为"xgrid_ygrid"
        
              xgrid = int(x) / grid_size
              ygrid = int(y) / grid_size

        Args:
            self : self对象
            grid_size : 网格的大小, int
        Return: 
            a tuple, (x_grid, y_grid)
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not grids")
        x_grid = int(self.x) / int(grid_size)
        y_grid = int(self.y) / int(grid_size)
        return (x_grid, y_grid)

    def destroy(self):
        """
        Desc: 销毁这个Point
              Python对象并没有销毁, 只是self.x, self.y == None
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.x = None
        self.y = None

    def is_empty(self):
        """
        Desc: 判断对象是否有效
        Args:
            self :
        Return: 
            None 
        Raise: 
            None
        """
        return self.x is None


class Segment(object):
    """
    Desc: 线段对象, 表示一个线段, 线段是无方向的
    """
    __slots__ = ("start", "end")

    def __init__(self, start, end):
        """
        Desc: 基于起终点坐标初始化线段
        Args:
            self : self
            start : 起点
            end : 终点
        Return: 
            None 
        Raise: 
            None
        """
        if start == end:
            raise error.RegionError("segemment __init__ start == end error")
        self.start = start
        self.end = end

    def __eq__(self, seg):
        """
        Desc: 判断两个seg是否是同一个seg
        Args:
            self : self
            seg : 另一个seg
        Return: 
            True, 两个seg相等
            False, 两个seg不相等
        Raise: 
            None
        """
        if seg is not None and ((self.start == seg.start and self.end == seg.end) \
           or (self.end == seg.start and self.start == seg.end)):
            return True
        return False

    def __str__(self):
        """
        Desc: 线段的字符串表达
        Args:
            self : self
        Return: 
            线段的字符串形式, "x1 y1,x2 y2"
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't str()")
        start = str(self.start)
        end = str(self.end)
        if start < end:
            retstr = "%s,%s" % (start, end)
        else:
            retstr = "%s,%s" % (end, start)
        return retstr
    
    def askey(self):
        """
        Desc: segment作为key的对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object hasn't key")
        if self.start.x < self.end.x \
                or (self.start.x == self.end.x and self.start.y < self.end.y):
            ret = (self.start.x, self.start.y, self.end.x, self.end.y)
        else:
            ret = (self.end.x, self.end.y, self.start.x, self.start.y)
        return ret

    def destroy(self):
        """
        Desc: 销毁seg对象, segment对象会变成一个空对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.start = self.end = None

    def is_empty(self):
        """
        Desc: 判断一个对象是否为空对象
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        return self.start is None

    def length(self):
        """
        Desc: 计算link的长度
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty Segment hasn't length")
        return pt_2_pt_dist(self.start, self.end)

    def linestring(self):
        """
        Desc: 将seg转换为spgeo.LineString
        
        Args:
            self : self
        Return: 
            spgeo.LineString
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't trans to spgeo.LineString")
        points = []
        points.append((self.start.x, self.start.y))
        points.append((self.end.x, self.end.y))
        return spgeo.LineString(points)

    def __grids(self, x, y, grid_size, grid_set):
        """
        Desc: 计算指定点的grid, 可能为多个
        Args:
            self : self
            x : x
            y : y
            grid_size : grid_size
            grid_set : 全局grid set
        Return: 
            list of (grid_x, grid_y)
            如果点在grid内, 返回一个grid; 
            如果点在grid边上, 返回两个grid;
            如果点在grid顶点上, 返回四个grid;
        Raise: 
            None
        """
        grid_x = int(x) / grid_size
        grid_y = int(y) / grid_size
        gs = [(grid_x, grid_y)]
        if grid_x * grid_size == x:
            gs.append((grid_x - 1, grid_y))
        if grid_y * grid_size == y:
            for i in range(len(gs)):
                gs.append((gs[i][0], grid_y - 1))
        for gd in gs:
            grid_set.add(gd)

    def grids(self, grid_size=1024):
        """
        Desc: 计算线段所涉及的grids
        Args:
            self : self
            grid_size : 网格大小, int
        Return: 
            [] grid list
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not grids")
        grid_set = set([])
        self.__grids(self.start.x, self.start.y, grid_size, grid_set)
        self.__grids(self.end.x, self.end.y, grid_size, grid_set)
        min_gx = int(min(self.start.x, self.end.x)) / grid_size
        max_gx = int(max(self.start.x, self.end.x)) / grid_size
        min_gy = int(min(self.start.y, self.end.y)) / grid_size
        max_gy = int(max(self.start.y, self.end.y)) / grid_size
        for grid_x in range(min_gx + 1, max_gx + 1):
            x = grid_x * grid_size
            y = self.start.y + (self.end.y - self.start.y) * (x - self.start.x) \
                    / (self.end.x - self.start.x)
            self.__grids(x, y, grid_size, grid_set)
        for grid_y in range(min_gy + 1, max_gy + 1):
            y = grid_y * grid_size
            x = self.start.x + (self.end.x - self.start.x) * (y - self.start.y) \
                    / (self.end.y - self.start.y)
            self.__grids(x, y, grid_size, grid_set)
        return list(grid_set)

    def intersect(self, seg):
        """
        Desc: 计算两个seg之间的交点, 正常情况下, 两个seg只有一个交点
              如果两个线段之间有重合的部分,则可能会有多个交点
        Args:
            self : self
            seg : 另一个seg
        Return: 
            list of points
        Raise: 
            None
        """
        thres = 0.001 / Point.base   # if t < thres, then we consider t = 0
        s01_x = self.end.x - self.start.x
        s01_y = self.end.y - self.start.y
        s23_x = seg.end.x - seg.start.x
        s23_y = seg.end.y - seg.start.y
        s20_x = self.start.x - seg.start.x
        s20_y = self.start.y - seg.start.y
        d01_len = math.sqrt(s01_x * s01_x + s01_y * s01_y)
        d23_len = math.sqrt(s23_x * s23_x + s23_y * s23_y)
        denom = s01_x * s23_y - s23_x * s01_y
        if denom == 0:  # 平行或共线
            d20_len = math.sqrt(s20_x * s20_x + s20_y * s20_y)
            cos201 = 0
            if d20_len > thres:
                cos201 = (s01_x * s20_x + s01_y * s20_y) / (d01_len * d20_len)
            sin201_2 = 1.0 - cos201 * cos201
            if sin201_2 < 0:
                sin201_2 = 0
            epson = d20_len * math.sqrt(sin201_2)
            if epson > 0.01 / Point.base:  # 平行, 不共线
                return []
            d1 = - d20_len * cos201
            if s01_x * s23_x + s01_y * s23_y > 0:
                d2 = d1 + d23_len
            else:
                d2 = d1 - d23_len
            d3 = d01_len
            if d2 < -thres and d1 < -thres or d2 > d3 + thres and d1 > d3 + thres:
                return []
            points = [(0, Point(self.start.x, self.start.y)), \
                      (d1, Point(seg.start.x, seg.start.y)), \
                      (d2, Point(seg.end.x, seg.end.y)), \
                      (d3, Point(self.end.x, self.end.y))]
            points.sort(key=lambda x: x[0])
            if points[1][1] == points[2][1]:
                return [points[1][1]]
            else:
                return [points[1][1], points[2][1]]
        s_numer = s01_x * s20_y - s01_y * s20_x
        t_numer = s23_x * s20_y - s23_y * s20_x
        s = s_numer / denom
        t = t_numer / denom
        if -s * d23_len > thres or (s - 1.0) * d23_len > thres:
            return []
        if -t * d01_len > thres or (t - 1.0) * d01_len > thres:
            return []
        if t >= 0 and t <= 1:
            if s < 0:
                return [Point(seg.start.x, seg.start.y)]
            if s > 1:
                return [Point(seg.end.x, seg.end.y)]
            else:
                return [Point(self.start.x + t * s01_x, self.start.y + t * s01_y)]
        if s >= 0 and s <= 1:
            if t < 0:
                return [Point(self.start.x, self.start.y)] 
            if t > 1:
                return [Point(self.end.x, self.end.y)]
        return []


class Region(object):
    """
    Desc: region对象, region对象是由一组segments组成
    """
    __slots__ = ("points", "holes")

    def __filter_points(self, points):
        """
        Desc: 过滤掉连续相同的点
        Args:
            self : self
            points : 原始点
        Return: 
            list of points, 过滤后的点list
        Raise: 
            None
        """
        if len(points) < 3:
            raise error.RegionError("points count < 3")
        pts = []
        lastpt = None
        for pt in points:
            if not pt == lastpt:
                pts.append(pt)
                lastpt = pt
        if pts[0] == pts[-1]:
            pts.pop()
        return pts

    def __init_points(self, points, holes=None):
        """
        Desc: 初始化
        Args:
            self : self
            points : 构成polygon的点, 首点和末点不相同
            holes : 区域内洞的表达
        Return: 
            None 
        Raise: 
            RegionError, when points is not valid
        """
        points = self.__filter_points(points)  # 删除重复点
        if len(points) < 3:
            raise error.RegionError("create polygon point count < 3")
        if not is_counter_clockwise(points):    # 外圈是逆时针
            self.points = points[::-1]
        else:
            self.points = points
        newholes = []
        if holes is not None and len(holes) > 0:
            for hole in holes:
                if len(hole) < 3:
                    raise error.RegionError("create polygon hole point count < 3")
                hole = self.__filter_points(hole)
                if is_counter_clockwise(hole):   # 内圈为顺时针
                    newholes.append(hole[::-1])
                else:
                    newholes.append(hole)
        self.holes = newholes

    def __init_segments(self, segments):
        """
        Desc: 基于segments创建region, 上一个segment的end是下一个segment的start
              最后一个segment的end = 第一个segment的start
              holes = None
        Args:
            self : self
            segments : 首尾相连的segment list
        Return: 
            None 
        Raise: 
            RegionError, when segments is not valid
        """
        if len(segments) < 3:
            raise error.RegionError("create polygon segment count < 3")
        points = [segments[0].start, segments[0].end]
        for seg in segments[1:]:
            if not seg.start == points[-1]:
                raise error.RegionError("create polygon neigbor segments is not valid")
            points.append(seg.end)
        if not points[0] == points[-1]:
            raise error.RegionError("create polygon first/last segments is not valid")
        self.__init_points(points)
    
    def __init_polygon(self, polygon):
        """
        Desc: 根据spgeo.Polygon生成region
        Args:
            self : self
            polygon : spgeo.Polygon
        Return: 
            None 
        Raise: 
            None
        """
        points = []
        for pt in polygon.exterior.coords[:-1]:
            points.append(Point(pt[0], pt[1]))
        holes = []
        for inner in polygon.interiors:
            hole_points = []
            for pt in inner.coords[:-1]:
                hole_points.append(Point(pt[0], pt[1]))
            holes.append(hole_points)
        self.__init_points(points, holes)

    def __init_wkt(self, wktstr):
        """
        Desc: 根据wkt字符串生成region
        Args:
            self : self
            wktstr : wkt字符串
        Return: 
            None 
        Raise: 
            RegionError
        """
        try:
            geo = wkt.loads(wktstr)
            if geo.geom_type == "Polygon":
                self.__init_polygon(geo)
            else:
                raise error.RegionError("It's not a polygon wkt string")
        except Exception as e: 
            raise error.RegionError("load from wkt str failed:" + e.message)

    def __init__(self, first, second=None):
        """
        Desc: 根据参数的类型构建region
        Args:
            self : self
            first : 第一个参数, 可以是list of Point, wkt str, polygon, list of Segment
            second : 第二个参数, 可以是holes, 仅当第一个参数是list of Point有效
        Return: 
            None 
        Raise: 
            None
        """
        if isinstance(first, list):
            if len(first) > 0:
                if isinstance(first[0], Point):
                    self.__init_points(first, second)
                elif isinstance(first[0], Segment):
                    self.__init_segments(first)
                else:
                    raise error.RegionError("can't init region")
        elif isinstance(first, spgeo.Polygon):
            self.__init_polygon(first)
        elif isinstance(first, str):
            self.__init_wkt(first)
        else:
            raise error.RegionError("can't init region")

    def __str__(self):
        """
        Desc: 将region转成一个字符串格式, 采用wkt格式
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object can't str()")
        ss = []
        for pt in self.points:
            ss.append(str(pt))
        ss.append(ss[0])
        wktstr = "POLYGON((" + ",".join(ss) + ")"
        if self.holes is not None:
            for hole in self.holes:
                ss = []
                for pt in hole:
                    ss.append(str(pt))
                ss.append(ss[0])
                wktstr += ",(" + ",".join(ss) + ")"
        wktstr += ")"
        return wktstr

    def assign(self, region):
        """
        Desc: region重新赋值
        Args:
            self : self
            region : 一个新的region
        Return: 
            None 
        Raise: 
            None
        """
        self.points = region.points
        self.holes = region.holes
    
    def __ring_2_segments(self, points):
        """
        Desc: 基于一个ring的points列表, 生成segments列表
        Args:
            self : self
        Return: 
            list of Segment
        Raise: 
            None
        """
        ret = []
        if points is None or len(points) < 2:
            return ret
        lastpt = points[-1]
        for pt in points:
            ret.append(Segment(lastpt, pt))
            lastpt = pt
        return ret

    def segments(self):
        """
        Desc: 获取region对应的所有segments
        Args:
            self : self
        Return: 
            list of Segment
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object hasn't segments")
        ret = self.__ring_2_segments(self.points)
        for hole in self.holes:
            segs = self.__ring_2_segments(hole)
            ret.extend(segs)
        return ret

    def destroy(self):
        """
        Desc: 销毁region, 变成一个空region
        Args:
            self : self
        Return: 
            None 
        Raise: 
            None
        """
        self.points = self.holes = None

    def is_empty(self):
        """
        Desc: 判断一个对象是否是空的
        Args:
            self :
        Return: 
            None 
        Raise: 
            None
        """
        return self.points is None

    def center(self):
        """
        Desc: 计算区域的中心点
        Args:
            self : self
        Return: 
            Point
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object has not center")
        pt = self.polygon().representative_point()
        return Point(pt.x, pt.y)

    def mbr(self):
        """
        Desc: 计算region的mbr
        Args:
            self : self
        Return: 
            [Point(minx, miny), Point(maxx, maxy)]
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not mbr()")
        maxx = minx = self.points[0].x
        maxy = miny = self.points[0].y
        for pt in self.points[1:]:
            if pt.x < minx:
                minx = pt.x
            elif pt.x > maxx:
                maxx = pt.x
            if pt.y < miny:
                miny = pt.y
            elif pt.y > maxy:
                maxy = pt.y
        return (Point(minx, miny), Point(maxx, maxy))

    def grids(self, grid_size=1024):
        """
        Desc: region所在的grids
        Args:
            self : self
            grid_size : grid_size
        Return: 
            list of grid, [(x_grid1, y_grid1), (x_grid2, y_grid2), ...]
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not grids()")
        bounds = self.mbr()
        start_grid = bounds[0].grid(grid_size)
        end_grid = bounds[1].grid(grid_size)
        ret = []
        for x_grid in range(start_grid[0], end_grid[0] + 1):
            for y_grid in range(start_grid[1], end_grid[1] + 1):
                ret.append((x_grid, y_grid))
        return ret

    def polygon(self):
        """
        Desc: 将region转换成spgeo.Polygon
        
        Args:
            self : self
        Return: 
            spgeo.Polygon
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not polygon()")
        points = []
        for pt in self.points:
            points.append((pt.x, pt.y))
        holes = []
        if self.holes is not None:
            for hole in self.holes:
                hole_points = []
                for pt in hole:
                    hole_points.append((pt.x, pt.y))
                holes.append(hole_points)
        return spgeo.Polygon(points, holes)
        
    def __operate(self, op, region):
        """
        Desc: 两个region进行交集, 差集, 并集运算
        
        Args:
            self : self
            op : "interact", "subtract", "union"
            region : 另一个region
        Return: 
            list of region
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not __operate()")
        ret = []
        polygon1 = self.polygon()
        polygon2 = region.polygon()
        if op == "intersect":
            inter = polygon1.intersection(polygon2)
        elif op == "subtract":
            inter = polygon1.difference(polygon2)
        elif op == "union":
            inter = polygon1.union(polygon2)
        else:
            raise error.RegionError("unknown operation")
        if inter.geom_type == "Polygon":
            ret.append(Region(inter))
        elif inter.geom_type == "MultiPolygon" or inter.geom_type == "GeometryCollection":
            for poly in inter:
                if poly.geom_type == "Polygon":
                    ret.append(Region(poly))
        return ret

    def intersect(self, region):
        """
        Desc: 计算两个region的交集
        Args:
            self : self
            region : 另一个region
        Return: 
            list of region, 对于不是多边形的结果, 丢弃, 只保留多边形的结果.
        Raise: 
            None
        """
        return self.__operate("intersect", region)

    def subtract(self, region):
        """
        Desc: self - region的差集
        Args:
            self : self
            region : 另一个region
        Return: 
            list of region, 只保留多边形的结果
        Raise: 
            None
        """
        return self.__operate("subtract", region)

    def union(self, region):
        """
        Desc: self - region的并集
        
        Args:
            self : self
            region : 另一个region
        Return: 
            list of region, 只保留多边形的结果
        Raise: 
            None
        """
        return self.__operate("union", region)

    def area(self):
        """
        Desc: 计算region的面积
        Args:
            self : self
        Return: 
            float, 面积
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not area()")
        ret = []
        polygon = self.polygon()
        return polygon.area

    def length(self):
        """
        Desc: 计算region的周长
        Args:
            self : self
        Return: 
            float, region的周长
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not length()")
        polygon = self.polygon()
        return polygon.length

    def gridize(self, grid_size=1024):
        """
        Desc: 网格化, 将一个多边形根据grid划分成多个多边形
        Args:
            self : self
            grid_size : 网格大小
        Return: 
            list of Region 
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not gridize()")
        points = []
        bounds = self.mbr()
        start_grid = bounds[0].grid(grid_size)
        end_grid = bounds[1].grid(grid_size)
        minx = (start_grid[0] - 1) * int(grid_size)
        maxx = (end_grid[0] + 2) * int(grid_size)
        miny = (start_grid[1] - 1) * int(grid_size) 
        maxy = (end_grid[1] + 2) * int(grid_size)
        use_min = True
        for x_grid in range(start_grid[0] + 1, end_grid[0] + 1):
            x = x_grid * int(grid_size)
            if use_min:
                points.append((x, miny))
                points.append((x, maxy))
            else:
                points.append((x, maxy))
                points.append((x, miny))
            use_min = not use_min
        if use_min:
            y_list = range(start_grid[1] + 1, end_grid[1] + 1)
        else:
            y_list = range(end_grid[1], start_grid[1], -1)
        use_max = True
        for y_grid in y_list:
            y = y_grid * int(grid_size)
            if use_max:
                points.append((maxx, y))
                points.append((minx, y))
            else:
                points.append((minx, y))
                points.append((maxx, y))
            use_max = not use_max
        ret = []
        if len(points) > 0:
            line = spgeo.LineString(points)
            polygons = spops.split(self.polygon(), line)
            for poly in polygons:
                if poly.geom_type == "Polygon":
                    ret.append(Region(poly))
        else:
            ret.append(self)
        return ret

    def contains(self, point):
        """
        Desc: 判断一个点是否在region内部
        Args:
            self : self
            point : 点
        Return: 
            True, 如果点在polygon内部或边缘
            False, 其它
        Raise: 
            None
        """
        if self.is_empty():
            raise error.RegionError("empty object have not contains()")
        return self.polygon().covers(point.point())


def is_counter_clockwise(points):
    """
    Desc: 判断points构建的ring的时钟方向
    Args:
        self : self
    Return: 
        True, if ring is counter clockwise, otherwise False
    Raise: 
        None
    """
    pts = []
    for pt in points:
        pts.append((pt.x, pt.y))
    ring = spgeo.LinearRing(pts)
    return ring.is_ccw


def pt_2_pt_dist(pt1, pt2):
    """
    Desc: 计算两个点之间的距离
    Args:
        pt1 : Point object
        pt2 : Point object
    Return: 
        float, distance
    Raise: 
        None
    """
    return pt1.point().distance(pt2.point())


def pt_2_seg_dist(pt, seg):
    """
    Desc: 计算点pt到线段seg的距离
    Args:
        pt : Point object
        seg : Segment object
    Returns:
        float, distance
    Raises:
        None
    """
    return pt.point().distance(seg.linestring())


def pt_2_reg_dist(pt, reg):
    """
    Desc: 计算点pt到region的距离
    
    Args:
        pt : Point object
        reg : Region object
    Return: 
        float, distance
        0: 点在region的边上
        >0: 点在region的外部
        <0: 点在region的内部
    Raise: 
        None
    """
    polygon = reg.polygon()
    point = pt.point()
    is_cover = polygon.covers(point)
    ring = spgeo.LinearRing(polygon.exterior.coords)
    min_dist = ring.distance(point)
    for hole in polygon.interiors:
        ring = spgeo.LinearRing(hole.coords)
        dist = ring.distance(point)
        if dist < min_dist:
            min_dist = dist
    if is_cover:
        return -min_dist
    else:
        return min_dist


def line_simp(points, threshold):
    """
    Desc: 将一组点进行douglas抽稀
    Args:
        points : list of mygeo.Point
        threshold : 抽稀的阈值
    Return: 
        list of new points
    Raise: 
        None
    """
    pts = []
    for pt in points:
        pts.append((pt.x, pt.y))
    line = spgeo.LineString(pts)
    newline = line.simplify(threshold, preserve_topology=False)
    newpoints = []
    for npt in newline.coords[:]:
        newpoints.append(Point(npt[0], npt[1]))
    return newpoints


def dump_segments(segments, seg_file):
    """
    Desc: 将segments转储到外存
    Args:
        segments : segments集合
        seg_file : seg_file
    Return: 
        None 
    Raise: 
        None
    """
    with open(seg_file, "w") as fo:
        for seg in segments:
            fo.write("%s\n" % (str(seg)))


def gen_segments(seg_file):
    """
    Desc: seg generator
    Args:
        seg_file : segment file
    Return: 
        None 
    Raise: 
        None
    """
    with open(seg_file, "rb") as fi:
        line = fi.readline()
        while line != "":
            p1, p2 = line.strip().split(",")
            x1, y1 = p1.split(" ")
            x2, y2 = p2.split(" ")
            yield Segment(Point(float(x1), float(y1)), Point(float(x2), float(y2)))
            line = fi.readline()


def load_segments(seg_file):
    """
    Desc: 从文件中加载segment list
    Args:
        seg_file : segment file
    Return: 
        None 
    Raise: 
        None
    """
    segs = []
    for seg in gen_segments(seg_file):
        segs.append(seg)
    return segs


def dump_regions(regions, reg_file):
    """
    Desc: 将regions转储到外存
    Args:
        regions : regions
        reg_file : region file
    Return: 
        None 
    Raise: 
        None
    """
    with open(reg_file, "w") as fo:
        for reg in regions:
            fo.write("%s\n" % (str(reg)))


def gen_regions(reg_file):
    """
    Desc: 从reg_file中生成regions
    Args:
        reg_file : region file
    Return: 
        None 
    Raise: 
        None
    """
    with open(reg_file, "rb") as fi:
        line = fi.readline()
        while line != "":
            yield Region(line.strip())
            line = fi.readline()


def load_regions(reg_file):
    """
    Desc: 加载reg_file中的信息到内存中
    Args:
        seg_file : seg_file
    Return: 
        None 
    Raise: 
        None
    """
    regions = []
    for reg in gen_regions(reg_file):
        regions.append(reg)
    return regions

