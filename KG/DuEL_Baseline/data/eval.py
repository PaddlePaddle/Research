# !/bin/env python
# -*- coding: utf-8 -*-
#####################################################################################
#
#  Copyright (c) CCKS 2020 Entity Linking Organizing Committee.
#  All Rights Reserved.
#
#####################################################################################
"""
@version 2020-03-30
@brief:
    Entity Linking效果评估脚本，评价指标Micro-F1
"""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import json
from collections import defaultdict


class Eval(object):
    """
    Entity Linking Evaluation
    """

    def __init__(self, golden_file_path, user_file_path):
        self.golden_file_path = golden_file_path
        self.user_file_path = user_file_path
        self.tp = 0
        self.fp = 0
        self.total_recall = 0
        self.errno = None

    def format_check(self, file_path):
        """
        文件格式验证
        :param file_path: 文件路径
        :return: Bool类型：是否通过格式检查，通过为True，反之False
        """
        flag = True
        for line in open(file_path):
            json_info = json.loads(line.strip())
            if 'text_id' not in json_info:
                flag = False
                self.errno = 1
                break
            if 'text' not in json_info:
                flag = False
                self.errno = 2
                break
            if 'mention_data' not in json_info:
                flag = False
                self.errno = 3
                break
            if not isinstance(json_info['text_id'], unicode):
                flag = False
                self.errno = 4
                break
            if not json_info['text_id'].isdigit():
                flag = False
                self.errno = 5
                break
            if not isinstance(json_info['text'], unicode):
                flag = False
                self.errno = 6
                break
            if not isinstance(json_info['mention_data'], list):
                flag = False
                self.errno = 7
                break
            for mention_info in json_info['mention_data']:
                if 'kb_id' not in mention_info:
                    flag = False
                    self.errno = 7
                    break
                if 'mention' not in mention_info:
                    flag = False
                    self.errno = 8
                    break
                if 'offset' not in mention_info:
                    flag = False
                    self.errno = 9
                    break
                if not isinstance(mention_info['kb_id'], unicode):
                    flag = False
                    self.errno = 10
                    break
                if not isinstance(mention_info['mention'], unicode):
                    flag = False
                    self.errno = 11
                    break
                if not isinstance(mention_info['offset'], unicode):
                    flag = False
                    self.errno = 12
                    break
                if not mention_info['offset'].isdigit():
                    flag = False
                    self.errno = 13
                    break
        return flag

    def micro_f1(self):
        """
        :return: float类型：精确率，召回率，Micro-F1值
        """
        # 文本格式验证
        flag_golden = self.format_check(self.golden_file_path)
        flag_user = self.format_check(self.user_file_path)
        # 格式验证失败直接返回None
        if not flag_golden or not flag_user:
            return None, None, None
        precision = 0
        recall = 0
        self.tp = 0
        self.fp = 0
        self.total_recall = 0
        golden_dict = defaultdict(list)
        for line in open(self.golden_file_path):
            golden_info = json.loads(line.strip())
            text_id = golden_info['text_id']
            text = golden_info['text']
            mention_data = golden_info['mention_data']
            for mention_info in mention_data:
                kb_id = mention_info['kb_id']
                mention = mention_info['mention']
                offset = mention_info['offset']
                key = '\1'.join(
                    [text_id, text, mention, offset],
                ).encode('utf8')
                # value的第二个元素表示标志位，用于判断是否已经进行了统计
                golden_dict[key] = [kb_id, 0]
                self.total_recall += 1

        # 进行计算
        for line in open(self.user_file_path):
            golden_info = json.loads(line.strip())
            text_id = golden_info['text_id']
            text = golden_info['text']
            mention_data = golden_info['mention_data']
            for mention_info in mention_data:
                kb_id = mention_info['kb_id']
                mention = mention_info['mention']
                offset = mention_info['offset']
                key = '\1'.join(
                    [text_id, text, mention, offset],
                ).encode('utf8')
                if key in golden_dict:
                    kb_result_golden = golden_dict[key]
                    if kb_id.isdigit():
                        if kb_id in [kb_result_golden[0]] and kb_result_golden[1] in [0]:
                            self.tp += 1
                        else:
                            self.fp += 1
                    else:
                        continue
                        # nil golden结果
                        nil_res = kb_result_golden[0].split('|')
                        if kb_id in nil_res and kb_result_golden[1] in [0]:
                            self.tp += 1
                        else:
                            self.fp += 1
                    golden_dict[key][1] = 1
                else:
                    self.fp += 1
        if self.tp + self.fp > 0:
            precision = float(self.tp) / (self.tp + self.fp)
        if self.total_recall > 0:
            recall = float(self.tp) / self.total_recall
        a = 2 * precision * recall
        b = precision + recall
        if b == 0:
            return 0, 0, 0
        f1 = a / b
        return precision, recall, f1


if __name__ == '__main__':
    # utf-8格式
    # 输入golden文件，预测文件
    eval = Eval('./basic_data/test_result.json', './generated/test_pred.json')
    prec, recall, f1 = eval.micro_f1()
    print prec, recall, f1
    if eval.errno:
        print eval.errno
