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
    out=p3ac-$test_file-Att$attention-$geohash-$prefix_word-$poi_att-$poi_word-$gpu

    cp $conf_name.template $conf 
    sed -i "s#<gpu>#$gpu#g" $conf
    #sed -i "s#<mode>#$mode#g" $conf
    sed -i "s#<attention>#$attention#g" $conf
    sed -i "s#<test_file>#$test_file#g" $conf
    sed -i "s#<geohash>#$geohash#g" $conf
    sed -i "s#<prefix_word>#$prefix_word#g" $conf
    sed -i "s#<poi_att>#$poi_att#g" $conf
    sed -i "s#<poi_word>#$poi_word#g" $conf
    return 0
}

function dy_train_pred() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m train 1>../tmp/$out-train.out 2>../tmp/$out-train.err
    #dy_pred $conf $out 
}


function dy_pred() {
    local conf=$1
    local out=$2
    if [ "$conf" == "" -o "$out" == "" ];then
        return 1
    fi
    sh run.sh -c $conf -m predict 1>../tmp/$out-pred.out 2>../tmp/$out-pred.err
    #$python ndcg.py ../tmp/$out-pred.out 1,3,10 > ../tmp/$out-pred.eval 2>&1
}

conf=""
out=""
run_type=$1

# 4 gpu cards parallel execute
model_opts_00="False True False None False"
model_opts_01="True True False cross False"
model_opts_02="True True False cross_dot_ffn True"
model_opts_03="True True False cross_dot_ffn True"

#in P4, 4 GPU cards
#train
if [ "$run_type" != "eval" ];then
for ((i=0; i<4; i++)); do
    eval opts=\$model_opts_0$i
    init_conf $i nn train $opts 
    dy_train_pred $conf $out &
    sleep 2
done
wait
fi

if [ "$run_type" != "train" ];then
#dev
for ((i=0; i<4; i++)); do
    eval opts=\$model_opts_0$i
    init_conf $i nn dev $opts 
    dy_pred $conf $out &
    sleep 2
done
wait

#test
for ((i=0; i<4; i++)); do
    eval opts=\$model_opts_0$i
    init_conf $i nn test $opts 
    dy_pred $conf $out &
    sleep 2
done
wait
fi
