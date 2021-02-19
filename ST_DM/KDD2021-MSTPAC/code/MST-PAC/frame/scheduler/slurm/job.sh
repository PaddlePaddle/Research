#!/bin/bash
function main() {
    mpirun hostname
    root_dir=`pwd`
    source "${root_dir}/conf/var_sys.conf"
    source "${root_dir}/.env"

    log_path=${root_dir}/logs
    mkdir -p $log_path
    if [[ ${slurm_cuda_lib_path} != "" ]]; then
        export LD_LIBRARY_PATH=${slurm_cuda_lib_path}:$LD_LIBRARY_PATH
    fi
    export PYTHONPATH=${PYTHONPATH}:${root_dir}

    sh ${root_dir}/frame/scheduler/slurm/afs_mount.sh 2>&1 >> ${log_path}/preprocess.log
    mpirun sh -x ${root_dir}/frame/scheduler/slurm/preprocess.sh  >> ${log_path}/preprocess.log

    if [[ -d ${root_dir}/tools/nccl_2.2.12-1+cuda9.0_x86_64/lib ]]; then
        export LD_LIBRARY_PATH=${root_dir}/tools/nccl_2.2.12-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH
    fi

    node_num=$(cat nodelist-${SLURM_JOB_ID} | wc -l)
    iplist=`cat nodelist-${SLURM_JOB_ID} | xargs | sed 's/ /,/g'`
    
    echo "node_num=${node_num}, iplist: ${iplist}" 2>&1 >> ${log_path}/preprocess.log
    if [[ ${data_location} == "local" ]]; then
        mpirun sh -x ${root_dir}/frame/scheduler/slurm/update_thread.sh >> ${log_path}/update_thread.log 
    fi

    #mpirun -x iplist=${iplist} -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} -x PYTHONPATH=${PYTHONPATH} sh ${root_dir}/frame/scheduler/slurm/train.sh &>$log_path/train.log
    mpirun -x iplist=${iplist} -x node_num=${node_num} sh ${root_dir}/frame/scheduler/slurm/train.sh 2>&1 >> $log_path/train.log
    sh -x ${root_dir}/frame/scheduler/slurm/postprocess.sh 2>&1 >> $log_path/postprocess.log
}

main "$@"
