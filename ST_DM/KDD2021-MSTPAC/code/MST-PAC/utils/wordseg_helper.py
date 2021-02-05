#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: wordseg_helper.py
Author: map(zhuoan@baidu.com)
Date: 2018/09/20 11:32:46
"""
import sys
import os
from ps_wordseg import wordseg

reload(sys)
sys.setdefaultencoding('gb18030')


MAX_TERM_COUNT = 1024

class WordSegment(object):
    """
    wordseg load & tokenization
    """
    def __init__(self):
        self.conf_handle = wordseg.scw_load_conf("./ps_wordseg/chinese_gbk/scw.conf")
        self.dict_handle = wordseg.scw_load_worddict("./ps_wordseg/chinese_gbk")
        self.result_handle = wordseg.scw_create_out(MAX_TERM_COUNT * 10)
        token_handle = wordseg.create_tokens(MAX_TERM_COUNT)
        self.token_handle = wordseg.init_tokens(token_handle, MAX_TERM_COUNT)

    def __del__(self):
        wordseg.destroy_tokens(self.token_handle)
        wordseg.scw_destroy_out(self.result_handle)
        wordseg.scw_destroy_worddict(self.dict_handle)
        wordseg.scw_destroy_conf(self.conf_handle)

    def tokenization(self, line, seg_type=2):
        """
        get one line segment res, in gb18030
        seg_type:
        0- basic
        1- phrase
        2- both
        """
        basic_res = []
        segment_res = []
        if len(line) <= 0:
            return basic_res, segment_res
        
        ret = wordseg.scw_segment_words(self.dict_handle, self.result_handle, line, len(line), 1)
        if ret < 0:
            sys.stderr.write('scw_segment_words() failed!\n')
            return basic_res, segment_res
        
        if seg_type == 0 or seg_type == 2:
            # basic res
            token_count = wordseg.scw_get_token_1(self.result_handle, wordseg.SCW_BASIC,
                    self.token_handle, MAX_TERM_COUNT)
            l = wordseg.tokens_to_list(self.token_handle, token_count)
            for i, token in enumerate(l):
                basic_res.append([token[7], token[1], i])
 
        if seg_type == 1 or seg_type == 2:
            # segment res
            token_count = wordseg.scw_get_token_1(self.result_handle, wordseg.SCW_WPCOMP,
                self.token_handle, MAX_TERM_COUNT)
            l = wordseg.tokens_to_list(self.token_handle, token_count)
            for i, token in enumerate(l):
                segment_res.append([token[7], token[1], i])

        return basic_res, segment_res

