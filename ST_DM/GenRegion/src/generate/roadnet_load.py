# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as numpy

# 清洗并使用纽约的数据
class Roadnet(object):
    """
    将纽约的数据清洗一下
    """
    def __init__(self, path):
        """
        Desc: 构造函数，读取成pandas DataFrame
        return:
            None
        Raise:
            IOError
        """
        try:
            self.data = pd.read_csv(path)
        except IOError:
            print('Wrong path')

    def to_seglists(self):
        """
        Desc: 把数据按照edge转换成lists of points
        return:
            lists of points
        Raise:
        None
        """
        n = self.data.shape[0]
        segs = {}
        for i in range(n):
            x = self.data.loc[i,"XCoord"]
            y = self.data.loc[i,"YCoord"]
            edge = self.data.loc[i,"EDGE"]
            if edge not in segs:
                segs[edge] = [(x,y)]
            else:
                segs[edge].append((x,y))
        return list(segs.values())

if __name__ == "__main__":
    from conf import Conf
    road = Roadnet(Conf.new_york_raw)
    segslists = road.to_seglists()

