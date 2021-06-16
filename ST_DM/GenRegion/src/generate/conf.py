# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: conf.py
Author: duanjianguo(duanjianguo@baidu.com)
Date: 2019/11/14 10:13:37
"""

class Conf(object):
    """
    Desc: 所有资源路径
    """
    # 名族列表
    # 各种level
    prov_level = 1
    city_level = 2
    city_other_level = 21
    county_level = 3
    town_level = 4
    shangquan_level = 41
    block_level = 5
    block_other_level = 51

    # Ming's experiment 
    exp_beijing_file = "../result/block_beijing"
    exp_ny_file = "../result/block_ny"
    exp_bj_file = "../result/block_bj" # result from online source
    exp_sh_file = "../result/block_shanghai"
    exp_cd_file = "../result/block_chengdu"
    exp_sz_file = "../result/block_shenzhen"
    exp_wh_file = "../result/block_wuhan"
    new_york_raw = "../data/NewYork_Edgelist.csv"
    beijing_raw = "../data/Beijing_Edgelist.csv"
    shanghai_raw = "../data/Shangai_Edgelist.csv"
    chengdu_raw = "../data/Chengdu_Edgelist.csv"
    shenzhen_raw = "../data/Shenzhen_Edgelist.csv"
    wuhan_raw = "../data/Wuhan_Edgelist.csv"

    # gridize
    grid_level1 = 65536
    grid_level2 = 8192
    grid_level3 = 1024
    # simp threshold
    min_poly_area = 100     # 低于此面积的polygon，丢弃
#    simp_threshold = 16     # 线段抽稀的阈值
    # block gen
    min_aoi_area = 1000      # 低于此面积的aoi，将不参与block的生成，但会作为aoi数据
    min_block_area = 10000   # block的最小 area
    min_block_width = 20     # block的最小 area / length
    clust_width = 25         # 聚类的半径
    block_simp_threshold = 8    # block的线段抽稀阈值
#    intersect_aoi_ratio = 0.8     # 交集的部分占aoi面积的比例，必须 > intersect_aoi_ratio
#    intersect_block_ratio = 0.6   # 交集的部分占block面积的比例，必须 > intersect_block_ratio
    
