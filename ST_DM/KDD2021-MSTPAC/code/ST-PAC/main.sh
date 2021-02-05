#!/bin/bash

function init_conf() {
    gpu=$1
    mode=$2
    test_file=$3
    attention=$4
    geohash=$5
    prefix_word=$6
    poi_att=$7
    poi_word=$8
    day_id=$9
    if [ $# -lt 2 ];then
        return 1
    fi

    conf_name=conf/poi_qac_personalized/poi_qac_personalized.local.conf
    conf=$conf_name.Att$attention-$geohash-$prefix_word-$poi_att-$poi_word-$day_id
    out=st-pac-$test_file-$gpu

    cp $conf_name.template $conf 
    sed -i "s#<attention>#$attention#g" $conf
    sed -i "s#<test_file>#$test_file#g" $conf
    sed -i "s#<geohash>#$geohash#g" $conf
    sed -i "s#<prefix_word>#$prefix_word#g" $conf
    sed -i "s#<poi_att>#$poi_att#g" $conf
    sed -i "s#<poi_word>#$poi_word#g" $conf
    sed -i "s#<day_id>#$day_id#g" $conf
    return 0
}

function dy_train() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m train 1>../tmp/logs/output/train/$out.out 2>../tmp/logs/error/train/$out.err
}


function dy_pred() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m predict 1>../tmp/logs/output/test/$out.out 2>../tmp/logs/error/test/$out.err
}

conf=""
out=""
run_type=$1

# model_opts_01="False True False None False True"
# model_opts_01="True True False cross False True"
model_opts_01="True True False cross_dot_ffn False True"


#train
if [ "$run_type" != "eval" ];then
eval opts=\$model_opts_01
init_conf 1 nn train $opts 
dy_train $conf $out 
sleep 2
wait
fi

if [ "$run_type" != "train" ];then
#dev
eval opts=\$model_opts_01
init_conf 1 nn dev $opts 
dy_pred $conf $out 
sleep 2
wait

#test
eval opts=\$model_opts_01
init_conf 1 nn test $opts 
dy_pred $conf $out 
sleep 2
wait
fi
