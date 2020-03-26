#!/bin/bash

cd `dirname $0`

version='v1.0.0'
target_file="dusql_trained_model_$version.tar.gz"
echo "coming soon..."
tar xzf $target_file
rm -rf $target_file

