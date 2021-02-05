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
    if [ $# -lt 2 ];then
        return 1
    fi

    conf_name=conf/poi_qac_personalized/poi_qac_personalized.local.conf
    conf=$conf_name.Att$attention-$geohash-$prefix_word-$poi_att-$poi_word
    out=mst-pac-$test_file-Att$attention-$geohash-$prefix_word-$poi_att-$poi_word-$gpu

    cp $conf_name.template $conf 
    sed -i "s#<attention>#$attention#g" $conf
    sed -i "s#<test_file>#$test_file#g" $conf
    sed -i "s#<geohash>#$geohash#g" $conf
    sed -i "s#<prefix_word>#$prefix_word#g" $conf
    sed -i "s#<poi_att>#$poi_att#g" $conf
    sed -i "s#<poi_word>#$poi_word#g" $conf
    return 0
}

function dy_train() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m train 1>../tmp/logs/output/train/$out-train.out 2>../tmp/logs/error/train/$out-train.err
}

function dy_pred() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m predict 1>../tmp/logs/output/test/$out-pred.out 2>../tmp/logs/error/test/$out-pred.err
}


conf=""
out=""
run_type=$1

# model_opts_02="False True False None False"
# model_opts_02="True True False cross False"
model_opts_02="True True False cross_dot_ffn False"

#train
eval opts=\$model_opts_02
init_conf 2 nn train $opts 
dy_train $conf $out 



