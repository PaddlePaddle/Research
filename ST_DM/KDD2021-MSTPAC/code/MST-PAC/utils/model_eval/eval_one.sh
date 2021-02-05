#!/bin/bash

shell_path=$(cd $(dirname "$0"); pwd)
PYTHON=/home/map/tools/jumbo/bin/python4.8

out_file=score.tmp
eval_dest=score.out

statistic() {
    cat $1 | awk -F'\t'  '{if (length($1)>0 && $3 != "nan") {print $0}}' | sort -t$'\t' -k 1,1 > $out_file 
    #get auc
    cut -f2,3 $out_file | ${shell_path}/calc_auc
}

statistic $1 1>$eval_dest 2>/dev/null
auc=`grep auc: $eval_dest | awk -F':' '{print $2}'`
echo $auc
