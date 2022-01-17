#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: config.py
func: 配置参数解析
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/19
"""

import argparse
import os
import copy
import yaml


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print("{}{} : ".format(delimiter * " ", k))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print("{}{} : ".format(delimiter * " ", k))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            print("{}{} : {}".format(delimiter * " ", k, v))
        if k.isupper():
            print(placeholder)


def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    print_dict(config)


class AttrDict(dict):
    """AttrDic类
    """
    def __getattr__(self, key):
        """ __getattr__"""
        return self[key]

    def __setattr__(self, key, value):
        """__setattr__"""
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        """__deepcopy__"""
        return copy.deepcopy(dict(self))


def create_attr_dict(yaml_config):
    """建立config字典
    """
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser("generic-multi-model-fuese train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    args = parser.parse_args()
    return args


def get_config(fname, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    if show:
        print_config(config)
    # check_config(config)
    return config

