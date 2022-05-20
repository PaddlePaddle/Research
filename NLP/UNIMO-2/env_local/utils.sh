#!/bin/bash

#set -u

function check_iplist() {

    if [ ${iplist:-} ]; then
        #paddle envs
        export PADDLE_PSERVER_PORT=9184
        export PADDLE_TRAINER_IPS=${iplist} 
        #export PADDLE_CURRENT_IP=`/sbin/ip a | grep inet | grep global | awk '{print $2}' | sed 's/\/[0-9][0-9].*$//g'`
        export PADDLE_CURRENT_IP=`hostname -i`
        
        iparray=(${iplist//,/ })
        for i in "${!iparray[@]}"; do
        if [ ${iparray[$i]} == ${PADDLE_CURRENT_IP} ]; then
            export PADDLE_TRAINER_ID=$i
        fi
        done
        
        export TRAINING_ROLE=TRAINER
        #export PADDLE_PSERVERS=127.0.0.1
        export PADDLE_INIT_TRAINER_COUNT=${#iparray[@]}
        export PADDLE_PORT=${PADDLE_PSERVER_PORT}
        export PADDLE_TRAINERS=${PADDLE_TRAINER_IPS}
        export POD_IP=${PADDLE_CURRENT_IP}
        export PADDLE_TRAINERS_NUM=${PADDLE_INIT_TRAINER_COUNT}
            #is model_config
        export PADDLE_IS_LOCAL=0
        echo "****************************************************"
  
        #paddle debug envs
        export GLOG_v=0
        export GLOG_logtostderr=1
        
        #nccl debug envs
        export NCCL_DEBUG=INFO
        #export NCCL_IB_DISABLE=1
        #export NCCL_IB_GDR_LEVEL=4
        export NCCL_IB_GID_INDEX=3
        #export NCCL_SOCKET_IFNAME=eth2
    fi
}

function ReadINIfile() {
    Section=$1
    Key=$2
    Configfile=$3
    ReadINI=`awk -F '=' '/\['$Section'\]/{a=1}a==1&&$1~/'$Key'/{print $2;exit}' $Configfile`
    echo "$ReadINI"
}

function get_gpu_id() {
    gpu_node=$1
    selected_gpus=""
    gpu_array=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15")
    for(( i=0;i<${gpu_node};i++ )) do
        if [[ ${selected_gpus} == "" ]]; then
                selected_gpus=${gpu_array[i]}
        else
                selected_gpus=${selected_gpus}","${gpu_array[i]}
        fi
    done;
    echo "${selected_gpus}"
}

function get_init_step() {
    init_model=$1
    cur_step=`basename $init_model`
    arr_cur_step=(${cur_step//_/ })
    if [[ ${arr_cur_step[0]} == "step" ]] && [[ ${#arr_cur_step[@]} == 2 ]]; then
        init_step=${arr_cur_step[1]}
    else
        init_step=0
    fi
    echo $init_step
}

function is_port_used() {
    port=$1

    ret=`lsof -i:$port`
    ret=$(echo $ret | grep "LISTEN")
    if [[ $ret != "" ]];
    then
        echo "1"
    else
        echo "0"
    fi
}

function is_ports_used() {
    port=$1
    nums=$2

    ret="0"
    for((i=0;i<${nums};i++)) do
        tmp=`is_port_used $[$port+$i]`
        if [[ ${tmp} == "1" ]]; then
            ret="1"
            break;
        fi
    done;
    echo $ret
}
