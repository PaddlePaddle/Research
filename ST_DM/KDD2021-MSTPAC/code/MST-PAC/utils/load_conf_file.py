#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: load_conf_file.py
Author: tuku(tuku@baidu.com)
Date: 2019/06/22 23:06:02
"""

import argparse
import os
import sys
import re
if sys.version_info >= (3, 2):
    import configparser as ConfigParser
else:
    import ConfigParser

from utils import flags


class ExtParser(ConfigParser.SafeConfigParser):
    """
    ExtParser for custom defined params, only PY2 need
    """
    def __init__(self, *args, **kwargs):
        self.cur_depth = 0
        #super(ExtParser, self).__init__(self, *args, **kwargs)
        ConfigParser.SafeConfigParser.__init__(self, *args, **kwargs)

    def get(self, section, option, raw=False, vars=None):
        r_opt = ConfigParser.SafeConfigParser.get(self, section, option, raw=True, vars=vars)
        if raw:
            return r_opt
        ret = r_opt
        re_oldintp = r'%\((\w*)\)s'
        re_newintp = r'\$\{(\w*):?(\w*)\}'

        m_new = re.findall(re_newintp, r_opt)
        if m_new:
            for f_section, f_option in m_new:
                no_sec = False
                if len(f_option) == 0:
                    f_option = f_section
                    f_section = "DEFAULT"
                    no_sec = True
                self.cur_depth = self.cur_depth + 1 
                if self.cur_depth < ConfigParser.MAX_INTERPOLATION_DEPTH:
                    sub = self.get(f_section, f_option, vars=vars)
                    if no_sec:
                        ret = ret.replace('${{{0}}}'.format(f_option), sub)
                    else:
                        ret = ret.replace('${{{0}:{1}}}'.format(f_section, f_option), sub)
                else:
                    raise ConfigParser.InterpolationDepthError(option, section, r_opt)

        m_old = re.findall(re_oldintp, r_opt)
        if m_old:
            for l_option in m_old:
                self.cur_depth = self.cur_depth + 1 
                if self.cur_depth < ConfigParser.MAX_INTERPOLATION_DEPTH:
                    sub = self.get(section, l_option, vars=vars)
                    ret = ret.replace('%({0})s'.format(l_option), sub)
                else:
                    raise ConfigParser.InterpolationDepthError(option, section, r_opt)

        self.cur_depth = self.cur_depth - 1 
        return ret 


def get_config_parser():
    """
        for py2 and py3
    """
    if sys.version_info >= (3, 2):
        conf_parser = ConfigParser.ConfigParser(interpolation=ConfigParser.ExtendedInterpolation())
    else:
        conf_parser = ExtParser()
    return conf_parser


class LoadConfFile(argparse.Action):
    """
    load conf file during arg_parser action
    """
    def __init__(self, option_strings, sec_name='DEFAULT', *args, **kwargs):
        self._sec_name = sec_name
        self._dest = None
        if 'dest' in kwargs:
            self._dest = kwargs['dest']
        super(LoadConfFile, self).__init__(
            option_strings=option_strings,
            *args, **kwargs)

    # Convert conf to a dict, and assign it to namespace 
    def __convert_to_dict(self, conf_parser):
        """
        convert config_parser to python dict
        """
        conf_dict = {}
        sections = conf_parser.sections()

        for section in sections:
            conf_dict[section] = {}
            for key, val in conf_parser.items(section):
                conf_dict[section][key] = val
        return conf_dict

    def __type_cast(self, value):
        """
        Type cast for config value.
            1. recognize the type of configure value automatically
            2. user can force convert the config value in conf with [[str|float|int]]
        """
        if value == 'None':
            return None
        type_dict = {'int': int, 'float': float, 'str': str, 'bool': bool}
        if value.startswith('[['):
            reg_value = re.findall('\[\[(.*)\]\](.*)', value)
            if len(reg_value) > 0:
                list_value = list(reg_value[0])
                #print list_value
                if len(list_value) > 1 and list_value[0] in type_dict:
                    return (type_dict[list_value[0]])(list_value[1])
        for convert in int, float, str:
            try:
                if value.lower() in ['true', 'false']:
                    return value.lower() == 'true'
                return convert(value)
            except:
                pass
        return value

    def __call__(self, arg_parser, namespace, value, option_string=None):
        """
        argparse action, set action as callable object
        """
        option_key = option_string.replace("--", "")
        namespace.__dict__[option_key] = value
        """
        if os.path.isfile(value): 
            fp = open(value)
        else:
            return
        """

        file_list = []
        if os.path.isfile(value) and (
            value.find("distributed.conf") or value.find("paddlecloud.conf")):
            additional_file = value.replace(
                "distributed.conf", "local.conf").replace(
                "paddlecloud.conf", "local.conf")
            if additional_file.endswith('.tmp'):
                additional_file = additional_file[:-4]
            if not os.path.isfile(additional_file):
                raise Exception("No local conf found:%s" % additional_file)
            
            file_list.append(additional_file)
        file_list.append(value)
        
        conf_parser = get_config_parser()
        #conf_parser.readfp(fp)
        conf_parser.read(file_list)

        # When sec_name is given, check if it exists
        if self._sec_name != 'DEFAULT' and not conf_parser.has_section(self._sec_name):
            raise ValueError('Invalid conf section name[%s]' % self._sec_name)


        conf_dict = self.__convert_to_dict(conf_parser)

        default_section_dict = conf_parser.defaults()

        param_section_dict = conf_dict.get(self._sec_name, {})
        for key in param_section_dict:
            value = conf_parser.get(self._sec_name, key)
            namespace.__dict__[key] = self.__type_cast(value)

        user_args_dict = conf_dict.get("USERARGS", {})
        for key in user_args_dict:
            if key not in conf_parser._sections["USERARGS"]:
                continue
            value = conf_parser.get("USERARGS", key)
            namespace.__dict__[key] = self.__type_cast(value)
        #fp.close()


def test(**kwargs):
    """
    print kwargs
    """
    print(kwargs)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("test_str", None, "test string")
    flags.DEFINE_boolean("test_bool", True, "test bool")
    flags.DEFINE_integer("test_int", None, "test int")
    flags.DEFINE_float("test_float", None, "test float")
    flags.DEFINE_custom("conf_file", "./conf/test/test.conf", 
        "conf file", action=LoadConfFile, sec_name="test")

    print("%s, %s" % (FLAGS.test_str, type(FLAGS.test_str)))
    print("%s, %s" % (FLAGS.test_bool, type(FLAGS.test_bool)))
    print("%s, %s" % (FLAGS.test_int, type(FLAGS.test_int)))
    print("%s, %s" % (FLAGS.test_float, type(FLAGS.test_float)))
    print("%s, %s" % (FLAGS.conf_file, type(FLAGS.conf_file)))
    test(**flags.get_flags_dict())
    #FLAGS.conf_file.seek(0)
    #print  FLAGS.conf_file.read()
