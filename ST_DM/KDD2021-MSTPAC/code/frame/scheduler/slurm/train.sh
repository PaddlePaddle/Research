#!/bin/bash 
set -x

main() {
    local root_dir="$(dirname $(readlink -f $0))/../../.."
    source ${root_dir}/conf/var_sys.conf
    source ${root_dir}/.env

    mkdir -p ${root_dir}/logs
   
    if [[ ${slurm_cuda_lib_path} != "" ]]; then
        export LD_LIBRARY_PATH=${slurm_cuda_lib_path}:$LD_LIBRARY_PATH
    fi
    if [[ -d ${root_dir}/tools/nccl_2.2.12-1+cuda9.0_x86_64/lib ]]; then
        export LD_LIBRARY_PATH=${root_dir}/tools/nccl_2.2.12-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH
    fi
    export PYTHONPATH=${PYTHONPATH}:${root_dir}

    conf_relative_name=$(echo $conf_file | awk -F"paddle-frame/" '{print $NF}')

    if [[ $iplist == "" ]]; then
        echo "[TRACE] $(date) iplist is null, set node_num as 1" >&2
        node_num=1
    fi
    
    cur_node_ip=$(hostname -i)
    echo "iplist is ${iplist} node_num is ${node_num} cur_node_ip: ${cur_node_ip} ${LD_LIBRARY_PATH} ${PYTHONPATH}" >&2

    local_tools_path=${root_dir}/tools
    fluid_bin=${local_tools_path}/paddle_release_home/python/bin/python

    #export GLOG_v=4
    #export GLOG_logtostderr=1
    if [[ $node_num -eq 1 ]]; then
        echo "[TRACE] $(date) One Node node_num is ${node_num}" >&2
        ${fluid_bin} -m paddle.distributed.launch ${root_dir}/frame/core/gpu_trainer.py \
                            --conf_file ${conf_relative_name} \
                            1>>${root_dir}/logs/train.log 2>>${root_dir}/logs/train.log.wf 
    else
        echo "[TRACE] $(date) Multi Node node_num is ${node_num}" >&2
        ${fluid_bin} -m paddle.distributed.launch --cluster_node_ips=${iplist} \
                            --node_ip=${cur_node_ip} \
                            ${root_dir}/frame/core/gpu_trainer.py \
                            --conf_file ${conf_relative_name} \
                            1>>${root_dir}/logs/train.log 2>>${root_dir}/logs/train.log.wf 

    fi
}

main "$@"
