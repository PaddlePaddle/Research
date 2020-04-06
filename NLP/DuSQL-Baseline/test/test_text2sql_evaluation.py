#!/usr/bin/env python3
# -*- coding:utf8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""unittest of .text2sql_evaluation as _eval
"""

import sys
import os
import traceback
import logging
import unittest
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..') + '/script')

import text2sql_evaluation as _eval

class TestText2SQLEvaluation(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tokenize(self):
        """test function of tokenize"""
        sql_list = [
                'select 书名, 类型 from 网络小说 where 评分 == ( select max ( 评分 ) from 网络小说 )',
                'select 书名 , 出版社 from 传记 where 作者 != "柳润墨" order by 页数 asc',
                '( select 书名 from 传记 where 页数 > 400 ) intersect ( select 书名 from 传记 where 出版时间 > "1981-03-24" )',
                '( select 姓名 from 作者 where 作品数量 >= 50 ) except ( select 姓名 from 作者 order by 出生日期 desc limit 3 )',
                '( select 开源课程名称 from 学校的开源课程 order by 课时 desc limit 3 ) except ( select 开源课程名称 from 学校的开源课程 where 主讲教师 != "王建安" )',
                'select avg ( 现价格 ) sum ( 原价格 ) from 本月特价书籍',
                'select max ( 电子书售价 ) from 电子书',
                'select min ( 电子书售价 ) avg ( 购买人数 ) max ( 会员价格 ) from 电子书',
                'select sum ( 豆瓣评分 ) max ( 1星占比 ) from 书籍',
                'select 出版社 from 文集 group by 出版社 order by avg ( 页数 ) desc limit 1',
                'select 名称 from 小说改编话剧 where 演出总场次 < ( select max ( 演出总场次 ) from 小说改编话剧 where 演出剧团 != "开心麻花" )',
                'select 名称 from 文集 where 页数 < ( select max ( 页数 ) from 文集 where 出版社 != "人民文学出版社" )',
                'select 名称 from 文集 where 页数 == ( select max ( 页数 ) from 文集 where 出版社 != "人民文学出版社" )',
                'select 名称 作者 from 书籍 where 豆瓣评分 > 5.4 order by 1星占比 desc',
                'select 名称 评价人数 * 1星占比 from 书籍 where 作者 == "塔拉·韦斯特弗"',
                'select 姓名 国籍 from 作者 where 作品数量 == ( select max ( 作品数量 ) from 作者 )',
                'select 讲述朝代 from 中国朝代历史 where COLUMN<>VALUE and COLUMN in(SELECT XXX)',
                'select 姓名 逝世日期-出生日期 from 作者 where 作品数量 < 50',
                'select 姓名 逝世日期 - 出生日期 from 作者 where 作品数量 < 50',
        ]
        for sql in sql_list:
            print(sql)
            print("_".join(_eval.tokenize(sql)))

if __name__ == "__main__":
    os.mkdir('log') if not os.path.isdir('log') else None
    logging.basicConfig(level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s %(filename)s'
        ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename='log/test_text2sql_evaluation.log',
        filemode='w')

    unittest.main()
