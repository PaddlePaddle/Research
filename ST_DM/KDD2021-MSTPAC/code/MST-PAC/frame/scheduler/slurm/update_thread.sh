#!/bin/bash
script_file=`readlink -f $0`
bindir=`dirname $script_file`

source $bindir/../conf/config.sh

had="hadoop fs -Dfs.default.name=$hdfs_data -Dhadoop.job.ugi=${hdfs_data_user},${hdfs_data_passwd}"

save_model_path=$bindir/../$local_model_path/$job_name

log_path=$bindir/../$local_log_path
mkdir -p $log_path
#gz fs -ls /user/lbs-mapsearch/rank-ltr/nn_se/dam/train_data.0403 | fgrep epoch | awk '{print $NF}' > o1
local_max_step=0
$had -test -z $hdfs_model_path
if [ $? -eq 0 ];then
    for ((i=0;i<5;++i))
    do
        $had -ls $hdfs_model_path | fgrep epoch | awk -F"/" '{print $NF}' |sed 's/epoch//g' | sort -k1nr | head -n 1 > $log_path/model_in_hdfs
        max_step=`cat $log_path/model_in_hdfs`
        if [[ $max_step != '' ]]; then
            break
        fi
        max_step=`$had -ls $hdfs_model_path | fgrep epoch | awk -F"/" '{print $NF}' |sed 's/epoch//g' | sort -k1nr | head -n 1`
        if [[ $max_step != '' ]]; then
            break
        fi
    done
else
    $had -mkdir $hdfs_model_path
    max_step=0
fi

function upload()
{
    local epoch=$1
    local hdfs=$2

    $had -test -z $hdfs/$epoch
    if [[ $? -eq 0 ]]; then
        return
    fi
    is_done=0
    for((i=0;i<5;++i))
    do 
        $had -mkdir $hdfs/$epoch
        $had -put $save_model_path/$epoch/* $hdfs/$epoch
        if [ $? -eq 0 ];then
            is_done=1
            break
        fi
    done
    
    if [ $is_done -eq 0 ];then
        $had -rmr $hdfs/$epoch
    else
        echo $epoch | sed 's/epoch//g' > $log_path/model_in_local
    fi
}
if [[ -e $log_path/model_in_local ]]; then
    local_max_step=`cat $log_path/model_in_local`
else
    local_max_step=0
fi
#max_step=20
for file in `ls -rtd $save_model_path/epoch*`
do
    file=`echo "$file" | sed 's/://g'`
    epoch=`echo "$file" | awk -F"/" '{print $NF}'`
    step=`echo "$epoch" | sed 's/epoch//g'`
    if [[ $max_step == '' ]]; then
        max_step=$local_max_step
    fi
    if [ $step -gt $max_step ];then 
        ## upload
        #echo $epoch
        upload $epoch $hdfs_model_path
        max_step=$step
    fi
done
