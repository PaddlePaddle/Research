#!/usr/bin/env bash

cd `dirname $0`

model_files_path="ernie1.0"

#get pretrained ernie1.0 model params
wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz
if [ ! -d $model_files_path ]; then
	mkdir $model_files_path
fi
tar xzf ERNIE_1.0_max-len-512.tar.gz -C $model_files_path
rm ERNIE_1.0_max-len-512.tar.gz

