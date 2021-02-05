#!/bin/bash

function preprocess_train_data() {
    echo "[Trace] $(date) start preprocess_local.." >&2
    #set in env file
    local local_data_path=${dataset_dir}

    mkdir -p ${local_data_path}

    hadoop fs -Dfs.default.name=$afs_host -Dhadoop.job.ugi=${afs_user},${afs_passwd} \
                                -get ${data_path}/*  $local_data_path
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Get train data ${data_path} failure." >&2
        exit 1
    fi

    echo "[Trace] $(date) preprocess_local success.." >&2
    return 0

}

function preprocess_tools_data() {
    echo "[Trace] $(date) Start preprocess_tools.." >&2

    hadoop fs -Dfs.default.name=$afs_host -Dhadoop.job.ugi=${afs_user},${afs_passwd} \
                                -mkdir ${afs_host}${output_path}

    hadoop fs -Dfs.default.name=$afs_host -Dhadoop.job.ugi=${afs_user},${afs_passwd} \
                                -get $fluid_afs  ${local_tools_path}
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Get afs fluid failure." >&2
        exit 1
    fi
    tar -zxf $local_tools_path/$(basename $fluid_afs)  -C $local_tools_path
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Tar fluid failure." >&2
        exit 1
    fi

    hadoop fs -Dfs.default.name=$afs_host -Dhadoop.job.ugi=${afs_user},${afs_passwd} \
                                -get $nccl_lib_afs  ${local_tools_path}
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Get nccl_lib_afs ${nccl_lib_afs} failure." >&2
        exit 1
    fi

    tar -zxf $local_tools_path/$(basename $nccl_lib_afs)  -C $local_tools_path
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Tar nccl_lib file failure." >&2
        exit 1
    fi

    echo "[Trace] $(date) preprocess_tools success.." >&2
    return 0
}

function main() {
    root_dir="$(dirname $(readlink -f $0))/../../../"
    source "${root_dir}/conf/var_sys.conf"
    source "${root_dir}/.env"

    local_tools_path=${root_dir}/tools/
    mkdir -p ${local_tools_path}

 
    #not afs mount mode. Get data from hdfs to each node.
    if [[ ${data_location} == "local" ]]; then
        preprocess_train_data
    fi
    preprocess_tools_data

    local local_model_path=${root_dir}/output
    mkdir -p  ${local_model_path}
    return 0
}

main "$@"
