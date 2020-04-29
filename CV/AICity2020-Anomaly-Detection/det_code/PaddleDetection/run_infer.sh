#!/bin/bash
start=$(date "+%s")
sh infer.sh dataset/track4/test_data_gt/test_data 0 ../test_output
now=$(date "+%s")
time=$((now-start))
echo "time used:$time seconds"
