#!/bin/bash
start=$(date "+%s")
sh infer.sh dataset/track4/track4_data_gt/test1/96 1 test &
sh infer.sh dataset/track4/track4_data_gt/test1/97 1 test &
sh infer.sh dataset/track4/track4_data_gt/test1/98 1 test &
sh infer.sh dataset/track4/track4_data_gt/test1/99 1 test &
sh infer.sh dataset/track4/track4_data_gt/test1/100 1 test 
now=$(date "+%s")
time=$((now-start))
echo "time used:$time seconds"
