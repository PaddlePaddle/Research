#!/bin/bash

function main() {
    echo "[Trace] $(date) start postprocess.." >&2
    root_dir="$(dirname $(readlink -f $0))/../../.."
    source "${root_dir}/.env" 
    if [[ -d ${root_dir}/output ]]; then
        if [[ $slurm_local_train_dir != "" ]]; then
            rm -r ${slurm_local_train_dir}/*
        fi
        mkdir -p ${slurm_local_train_dir}
        mv ${root_dir}/output/* ${slurm_local_train_dir}
        if [[ $? != 0 ]]; then
            echo "[FATAL] $(date) mv ${root_dir}/output to  \
                  ${slurm_local_train_dir} failure. cur dir $(pwd)" >&2
            exit 1
        fi
    else
        echo "[FATAL] $(date) generate ${root_dir}/output/ failure." >&2
        exit 1
    fi 

    echo "[Trace] $(date) end postprocess.." >&2
}

main "$@"
