#!/bin/bash

cd `dirname $0`

version='v1.0.0'
target_file="dusql_data_$version.tar.gz"
echo "coming soon..."
tar xzf $target_file
rm -rf $target_file

