#! /bin/env python
# encoding=utf-8
# 
#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief kvdict.py
 Author: map(zhuoan@baidu.com)
 Date: 2019/08/26 13:47:48
"""

import c_kvdict

class KVDict(object):
    """
        kvdict for python large mem
    """

    def __init__(self):
        self.__dict_handle = c_kvdict.create()
    
    def __del__(self):
        c_kvdict.release(self.__dict_handle)

    def load_files(self, flist, key_count, in_mem):
        """
            load files to kv
        """
        #change a filename to a list.
        if not isinstance(flist, list):
            flist = [flist]
        fcnt = len(flist)
        c_kvdict.load_files(self.__dict_handle, fcnt, flist, key_count, in_mem)

    def find(self, key):
        """
           find key 
        """
        return c_kvdict.find(self.__dict_handle, key)

    def has(self, key):
        """
           has key 
        """
        return c_kvdict.has(self.__dict_handle, key)
    
    def load_index_and_files(self, index_file, file_list, in_mem):
        """
            load index 
        """
        # change a filename to a list.
        if not isinstance(file_list, list):
            file_list = [file_list]
        fcnt = len(file_list)
        return c_kvdict.load_index_and_files(
                self.__dict_handle, index_file, len(file_list), file_list, in_mem)

    def write_index(self, output_file):
        """
           write index 
        """
        return c_kvdict.write_index(self.__dict_handle, output_file)

