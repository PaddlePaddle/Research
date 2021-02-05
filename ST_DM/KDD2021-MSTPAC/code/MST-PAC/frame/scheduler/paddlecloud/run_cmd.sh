#!/bin/bash
set -x
root_dir=$(cd $(dirname "$0")/../../../; pwd)
echo $root_dir
source ${root_dir}/env_paddlecloud
function raw_afs_mount() {
    fs_name=$1
    username=$2
    password=$3
    afs_local_mount_point=$4
    afs_remote_mount_point=$5

    mkdir -p ${afs_local_mount_point}

    pushd /opt/afs_mount
    nohup ./bin/afs_mount \
                --username=${username} \
                --password=${password} ${afs_local_mount_point} ${fs_name}${afs_remote_mount_point} 1>my_mount.log 2>&1 &
    popd
}

function run_train() {
    conf_file=$1
    mode=$2
    thirdparty_paddle_version=$3
    version=${thirdparty_paddle_version: 2: 1}
    
    if [ "$mode" == "cpu" ];then
        python_bin=./thirdparty/paddle_cpu/bin/python4.8
        #-m paddle.distributed.launch_ps
        $python_bin frame/core/paddlecloud_fleet_trainer.py --conf_file ${conf_file}
    else
        python_bin=${paddlecould_fluid_bin}
        if [ ${version} -ge 7 ]
        then
            $python_bin -m paddle.distributed.launch --use_paddlecloud frame/core/gpu_trainer.py --conf_file ${conf_file}
        else
            $python_bin -m paddle.distributed.launch --use_paddlecloud=True frame/core/gpu_trainer.py --conf_file ${conf_file}
        fi
    fi
}

function run_evaluate() {
    conf_file="$1"
    #change conf file to local
    local_conf_tmp="${conf_file/paddlecloud/local}"
    local_conf_tmp="${local_conf_tmp/distributed/local}"
    local_conf=${local_conf_tmp%.tmp}
    sed -i "s#^fluid_bin=.*#fluid_bin=${paddlecould_fluid_bin}#g" ./conf/var_sys.conf
    sed -i 's#^train_dir:.*#train_dir: output#g' $local_conf
    sed -i 's#^dataset_dir:.*#dataset_dir: ./afs_eval#g' $local_conf

    sh run.sh -c $local_conf -m monitor
}

function main() {
    set -x

    conf_file="$1"
    mode="$2"
    need_evaluate="$3"
    thirdparty_paddle_version="$4"
    # mount py runtime, faster than hadoop get
    raw_afs_mount ${fs_name} ${fs_ugi%,*} ${fs_ugi#*,} "/root/paddlejob/workspace/env_run/afs_runtime/" "/user/lbs-mapsearch/research/sunyibo/paddle_frame/thirdparty"
    sleep 2s
    tree -L 2 "/root/paddlejob/workspace/env_run/afs_runtime/"

    mkdir "py_runtime"
    tar xzf "./afs_runtime/paddle1.8_py3.8.tar.gz" -C "py_runtime"

    run_train ${conf_file} ${mode} ${thirdparty_paddle_version} &
    train_pid=$!
    if [[ ${eval_data_dir} ]] && [[ -n ${conf_file} ]] && [[ ${need_evaluate} -eq 1 ]];then
        afs_local_mount_point="/root/paddlejob/workspace/env_run/afs_eval/"
        raw_afs_mount ${fs_name} ${fs_ugi%,*} ${fs_ugi#*,} ${afs_local_mount_point} ${eval_data_dir}
        sleep 2s
        tree -L 2 ${afs_local_mount_point}
        # seperate monitor log
        run_evaluate ${conf_file} > ${root_dir}/../log/paddle_frame_monitor.log 2>&1 &
        eval_pid=$!
    fi
    wait $train_pid
    ret=$?
    wait $eval_pid
    return $ret
}

main "$@"
