#!/bin/bash


root_dir=$(cd $(dirname "$0")/../../; pwd)

# environment variables for fleet distribute training
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=2
export PADDLE_PORT=36011
export PADDLE_PSERVERS=127.0.0.1
export POD_IP=127.0.0.1
export OUTPUT_PATH="logs"
export SYS_JOB_ID="local_cluster"

export PADDLE_PSERVER_PORTS=36011
export PADDLE_PSERVER_PORT_ARRAY=(36011)

export PADDLE_PSERVER_NUMS=1
export PADDLE_TRAINERS=2

function main() {
    config_file="$1"
    source ${root_dir}/conf/var_sys.conf
    source ${root_dir}/.env 
    
    echo "WARNING: This script only for run PaddlePaddle Fluid on one node" >&2
    echo "Running 1X2 Parameter Server model" >&2

    export GLOG_v=0
    export GLOG_logtostderr=1
    #start ps
    export TRAINING_ROLE=PSERVER
    for((i=0; i<$PADDLE_PSERVER_NUMS; i++))
    do
        cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
        echo "PADDLE WILL START PSERVER "$cur_port >&2
        export PADDLE_TRAINER_ID=$i
        $fluid_bin frame/core/paddlecloud_fleet_trainer.py --conf_file ${config_file}  &> ./logs/pserver.$i.log &
    done
    
    sleep 3
    
    #start worker
    export TRAINING_ROLE=TRAINER
    for((i=0; i<$PADDLE_TRAINERS; i++))
    do
        echo "PADDLE WILL START Trainer "$i >&2
        export PADDLE_TRAINER_ID=$i
        $fluid_bin frame/core/paddlecloud_fleet_trainer.py --conf_file ${config_file} &> ./logs/trainer.$i.log &
    done

    wait
}

main "$@"
