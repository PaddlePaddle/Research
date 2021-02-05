#!/bin/bash

function preprocess_afs_mount() {
    echo "[Trace] $(date) Start preprocess_afs_mount.." >&2
    local local_mount_name=${mount_local_dir}

    hadoop fs -Dfs.default.name=$afs_host -Dhadoop.job.ugi=${afs_user},${afs_passwd} \
                                -mkdir ${afs_host}${output_path}

    sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh \
            ${afs_user} ${afs_passwd} \
            ${root_dir}${local_mount_name} \
            ${afs_host}${afs_mount_path}
    if [[ $? != 0 ]]; then
        echo "[FATAL] $(date) afs_mount ${root_dir}${local_mount_name} failure." >&2
        exit 1
    fi

    echo "[Trace] $(date) preprocess_afs_mount success.." >&2
    return 0
}

function main() {
    root_dir="$(dirname $(readlink -f $0))/../../../"
    source "${root_dir}/conf/var_sys.conf"
    source "${root_dir}/.env"
  
    preprocess_afs_mount 
}

main "$@"
