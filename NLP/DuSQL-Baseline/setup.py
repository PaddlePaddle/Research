#!/usr/bin/env python
# -*- coding:gb18030 -*-
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

"""setup file
"""

import sys
import os
import traceback
import logging
try:
    import ConfigParser as configparser
except:
    import configparser
import re
from collections import defaultdict
from distutils.util import convert_path
from fnmatch import fnmatch
from fnmatch import fnmatchcase
logging.basicConfig(level=logging.ERROR, datefmt='%m-%d %H:%M:%S', filename=None)

from setuptools import setup
from setuptools import find_packages

def read_setup_cfg(setup_cfg_file, section):
    """read setup cfg file
    
    Args:
        setup_cfg_file      setup.cfg file path
        section             section you want to read

    Returns:
        a dict: dict of all options in [$section] section
        None: some error happens

    Raises:
        None
    """
    try:
        # read config file
        config = configparser.ConfigParser()
        config.read(setup_cfg_file)
        # get metadata section
        metadata = config.items(section)
        # parse metadata list to dict and return
        return dict(metadata)
    except configparser.ParsingError as pe:
        logging.warning("%s", traceback.format_exc())
        return None
    except configparser.NoSectionError as nse:
        logging.warning("%s", traceback.format_exc())
        return None
    except configparser.Error as err:
        logging.warning("%s", traceback.format_exc())
        return None
    except Exception as e:
        logging.warning("%s", traceback.format_exc())
        return None


def metadata_of_setup_cfg(setup_cfg_file="setup.cfg"):
    """read [metadata] section of setup_file
    
    Args:
        setup_cfg_file: setup.cfg file path, default is setup.cfg is current path.

    Returns:
        a dict: dict of all options in [metadata] section
        None: some error happens

    Raises:
        None
    """

    return read_setup_cfg(setup_cfg_file, "metadata")


def install_requires_of_setup_cfg(setup_cfg_file="setup.cfg"):
    """read install_requires from setup cfg file
    
    Args:
        setup_cfg_file: setup.cfg file path, default is setup.cfg is current path.

    Returns:
        list        list of install_requires in setup.cfg
                    an empty list will be returned if some error happens

    Raises: None
    """
    requires = read_setup_cfg(setup_cfg_file, "requires")
    if requires is None or "install_requires" not in requires:
        return []

    install_requires = requires["install_requires"]
    return install_requires.strip().split('\n')


def list_files(path, pat):
    """list all files in path
    
    Args:
        path
        pat

    Returns:
        list 
    Raises: None
    """
    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        logging.warning("path [%s] is not exist.", path)
        return []

    hidden_file_patt = re.compile(r'(^\.|\/\.)[^\/]+')
    fn_filter_file = lambda x: any(fnmatch(x, p) for p in pat)
    result = []
    for root, dirs, files in os.walk(path):
        if re.search(hidden_file_patt, root) is not None:
            continue
        for filename in files:
            if re.search(hidden_file_patt, filename) is not None:
                continue
            fullpath = os.path.join(root, filename)
            if fullpath.startswith('./'):
                fullpath = fullpath[2:]
            if fn_filter_file(fullpath):
                result.append(fullpath)
    return result


def parse_include_files(dict_result, module_name, path, pat=('*')):
    """parse_include_files
    
    Args:
        dict_result     [in/out] defaultdict(list). 
        module_name     module name of caller of this function
        path            path that setted in MANIFEST.in after the inc_cmd
        pat             patten for recursive-include, default is ('*')

    Returns:
        defaultdict     key is a string of dest_dir; value is a list of file
                        pathes like ["local_file1", "local_file2"]

    Raises: None
    """
    lst_files = list_files(path, pat)
    for fullpath in lst_files:
        dirname = os.path.dirname(fullpath)
        filename = os.path.basename(fullpath)
        if filename in ('setup.cfg', 'MANIFEST.in', "README", "ChangeLog"):
            dest_dir = os.path.join('conf', module_name)
        elif (dirname.startswith('conf') or dirname.startswith('data'))\
                and (len(dirname) == 4 or dirname[4] == '/'):
            root_dir = dirname[:4]
            base_dir = dirname[5:]
            dest_dir = os.path.join(root_dir, module_name, base_dir)
        else:
            dest_dir = os.path.join('data', module_name, dirname)
        dest_dir = dest_dir.rstrip('/')
        dict_result[dest_dir].append(fullpath)

    return dict_result


def data_files_of_manifest(module_name, manifest_file="MANIFEST.in"):
    """data_files_of_manifest
    
    Args:
        module_name     module name of caller of this function
        manifest_file   default is "MANIFEST.in"

    Returns:
        list            data files of "data_files" format of setuptools.setup().
                        each item is a tuple with 2 items, like 
                        ("dest_dir", ["local_file1", "local_file2"]).
                        an empty list will be returned if some error happens
    Raises: None
    """
    if not os.path.isfile(manifest_file):
        logging.warning("%s is not a file", manifest_file)
        return []

    dict_result = defaultdict(list)
    with open(manifest_file) as ifs:
        split_patt = re.compile(r' +')
        for line in ifs:
            if line.lstrip().startswith('#'):
                continue
            fields = re.split(split_patt, line.strip(), 1)
            if len(fields) != 2:
                logging.warning("file %s line error: %s", manifest_file, line)
                continue
            cmd = fields[0]
            files = fields[1]
            if cmd == 'include':
                parse_include_files(dict_result, module_name, files)
            elif cmd == 'recursive-include':
                lst_file = re.split(split_patt, files)
                base = lst_file[0]
                pat = [os.path.join(base, x) for x in lst_file[1:]]
                parse_include_files(dict_result, module_name, base, pat)
            else:
                logging.warning("include command error: %s", cmd)
    return dict_result.items()

def update_version(step=1, setup_file="setup.cfg"):
    """read setup_file, get item version of [metadata] section,
    and add 1 to last last level of the version number
    
    Args:
        step: version add step, default is 1.
        setup_file: setup.cfg file path, default is setup.cfg is current path

    Returns:
        (old_version, new_version): it's a tuple
        None: some error happens

    Raises:
        None
    """

    try:
        # read config file
        config = configparser.ConfigParser()
        config.read(setup_file)

        curr_version = config.get("metadata", "version")
        version_list = curr_version.split('.')
        version_list.reverse()
        for idx, number in enumerate(version_list):
            if number.isdigit():
                version_list[idx] = str(int(number) + step)
                break
        version_list.reverse()
        new_version = '.'.join(version_list)
        if curr_version == new_version:
            return None

        # update version in setup.cfg file
        config.set("metadata", "version", new_version)
        config.write(open(setup_file, "r+"))
        return (curr_version, new_version)
    except configparser.ParsingError as pe:
        logging.error("%s", traceback.format_exc())
        return None
    except configparser.NoSectionError as nse:
        logging.error("%s", traceback.format_exc())
        return None
    except configparser.Error as err:
        logging.error("%s", traceback.format_exc())
        return None
    except Exception as e:
        logging.error("%s", traceback.format_exc())
        return None


def rollback_version(step=-1, setup_file="setup.cfg"):
    """call update_version, but default step is -1
    
    Args:
        step: version add step, default is 1.
        setup_file: setup.cfg file path, default is setup.cfg is current path

    Returns:
        (old_version, new_version): it's a tuple
        None: some error happens

    Raises:
        None
    """
    return update_version(step, setup_file)


class ScriptFinder(object):
    """scripts finder tool
    """

    EXCLUDE_DEFAULT = ()
    INCLUDE_DEFAULT = ('*script/*.py', '*module/*.py', '*script/*.sh', '*module/*.sh')
    @classmethod
    def find(cls, where='.', exclude=None, include=None):
        """do find script in $where directory, excluding $exclude,
        and including $include.
        
        Args:
            where       default is '.'
            exclude     default is ()
            include     default is ('*script?/*.py', '*module?/*.py', '*script?/*.sh', '*module?/*.sh')

        Returns:
            list            list of python file looks like scripts

        Raises: None
        """
        if exclude is None:
            exclude = cls.EXCLUDE_DEFAULT
        if include is None:
            include = cls.INCLUDE_DEFAULT

        out = cls._find_all_script_files(convert_path(where))
        fn_includes = cls._build_filter(*include)
        fn_exclues = cls._build_neg_filter('*setup.py', '__init__.py', *exclude)
        out = filter(fn_includes, out)
        out = filter(fn_exclues, out)
        return list(out)

    @classmethod
    def _find_all_script_files(cls, base_path):
        """find all python and shell files
        
        Args:
            base_path
        Returns:
            list 
        Raises: None
        """
        files = cls._all_script_files_iter(base_path)
        suitable = filter(
                lambda x: not os.path.isfile(os.path.join(os.path.dirname(x), '__init__.py')),
                files)
        return suitable

    @classmethod
    def _all_script_files_iter(cls, base_path):
        """find all python and shell files, relative to base_path
        
        Args:
            base_path   base path to find

        Returns:
            itertor     python and shell files
        Raises: None
        """
        exclude_dir = cls._build_filter('*.svn*', '*.git*')
        include_file = cls._build_filter('*.py', '*.sh')
        for root, dirs, files in os.walk(base_path, followlinks=True):
            if exclude_dir(root):
                continue
            for filename in files:
                if include_file(filename):
                    yield os.path.join(root, filename)

    @staticmethod
    def _build_filter(*patterns):
        """copy from setuptools/__init__.py:
        Given a list of patterns, return a callable that will be true only if
        the input matches one of the patterns.
        """
        return lambda name: any(fnmatchcase(name, pat=pat) for pat in patterns)

    @staticmethod
    def _build_neg_filter(*patterns):
        """copy from setuptools/__init__.py:
        Given a list of patterns, return a callable that will be true only if
        the input matches one of the patterns.
        """
        return lambda name: not any(fnmatchcase(name, pat=pat) for pat in patterns)


find_scripts = ScriptFinder.find


def extension_args():
    """return extension_args"""
    return {}


def rollback_version_wrapper(update_ret):
    """wrapper of rollback_version
    
    Args:
        update_ret: return value of update version

    Returns: None
    """
    if update_ret is not None:
        rollback_version()


def update_version_and_ci():
    """auto update version and process ci
    
    Args: None

    Exit:
        0: succ
        -1: update version failed
        1: ci failed
    """
    update_version_ret = update_version()
    if update_version_ret is None:
        sys.stderr.write("update version failed\n")
        exit(-1)
    sys.stderr.write("version was changed from %s to %s\n" % update_version_ret)
    user_input = raw_input("continue to ci? (Y/n): ")
    if not(user_input == '' or user_input.lower() == 'y'):
        rollback_version_wrapper(update_version_ret)
        sys.stderr.write("do nothing. goodbye~~\n")
        exit(0)
    os.system("git add setup.cfg")
    ret = os.system("git commit --amend --no-edit" % (' '.join(sys.argv[2:])))
    if ret == 0:
        sys.stderr.write("commit succ\n")
        exit(0)
    else:
        sys.stderr.write("commit failed!\n")
        rollback_version_wrapper(update_version_ret)
        exit(1)


# auto update version and ci
if len(sys.argv) >= 2 and sys.argv[1] == "ci":
    update_version_and_ci()

# get metadata from setup.cfg and process setup
metadata = metadata_of_setup_cfg()
if metadata is None:
    sys.stderr.write("read setup.cfg failed!\n")
    exit(-1)
module_name = metadata["name"]

# 用于扩展
metadata.update(extension_args())

# 为upload临时修改HOME环境变量，以便程序使用 virtual env 的pypirc
if "upload" in sys.argv and os.getenv("VIRTUAL_ENV") is not None:
    os.environ["HOME"] = os.getenv("VIRTUAL_ENV")
setup(
    # 模块的库文件，会被安装在python的lib中
    packages = find_packages(),
    # 模块的命令文件，会被安装在bin目录中
    scripts = find_scripts(),
    # 依赖的模块
    install_requires = install_requires_of_setup_cfg(),
    # 数据和配置
    data_files = data_files_of_manifest(module_name),

    # setup.cfg中配置的metadata
    **metadata
)

