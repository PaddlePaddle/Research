#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Setup script.

Authors: sunmingming01(sunmingming01@baidu.com)
Date:    2020/12/31 12:33:34
"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='paddlemetrics',
    version='1.0.0-beta',
    description='Metrics library for paddle, porting from torch metrics.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='Apache',
    packages=find_packages(include=["paddlemetrics", "paddlemetrics.*"]),
    author='Mingming Sun',
    author_email='sunmingming01@baidu.com',
    keywords=['Deep Learning', 'Paddlepaddle'],
    url='',
    download_url=''
)

install_requires = [
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)