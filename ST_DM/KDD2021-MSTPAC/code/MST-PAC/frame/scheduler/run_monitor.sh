#!/bin/bash 
root_dir=$(cd $(dirname "$0")/../../; pwd)

source "${root_dir}/frame/scheduler/scheduler_functions.sh"

function train_monitor() {
    config_file="$1"
    config_gpus="$2"
    config_gpu_num="$3"
    mode="$4"
    sleep_time=10

    local used_ckpt=""
    while true
    do
        train_dir_path=$(parse_user_conf $config_file "train_dir" "Train")
        #eval_cols=$(parse_user_conf ${config_file} "eval_cols" "Evaluate")
        ckpt_file="${train_dir_path}/checkpoint.meta"
        if [[ ! -e ${ckpt_file} ]]; then
            echo "[TRACE] $(date) file: ${ckpt_file} does not exist" >&2
            sleep ${sleep_time}
            continue
        fi
        sec_name="Monitor"
        local last_ckpt=$(parse_user_conf ${ckpt_file} "init_pretrain_model" ${sec_name})
        local ckpt_version=$(parse_user_conf ${ckpt_file} "ckpt_version" ${sec_name})
        local run_state=$(parse_user_conf ${ckpt_file} "run_state" ${sec_name})
        echo "[TRACE] $(date) last checkpoint is ${last_ckpt}" >&2
        if [ "$used_ckpt" == "$last_ckpt" ]; then
            echo "[TRACE] $(date) ckpt_version: ${ckpt_version}; last_ckpt: ${last_ckpt}; used_ckpt: ${used_ckpt}; sleep"  >&2
        else
            echo "[TRACE] $(date) ckpt_version: ${ckpt_version}; last_ckpt: ${last_ckpt}; used_ckpt: ${used_ckpt}; NEW"  >&2
            infer_data="${train_dir_path}/${ckpt_version}.pred"

            # TODO: framework bug, this is an ad-hoc fix. need re-design.
            origin_conf=${config_file%.tmp}
            cp ${origin_conf} ${origin_conf}.tmp

            sh -x ${root_dir}/frame/scheduler/run_predict.sh ${config_file} "${config_gpus}" "${config_gpu_num}" "${mode}" ${ckpt_file} 1>${infer_data}
            if [[ -e ${root_dir}/utils/evaluate.sh ]]; then
                sh -x ${root_dir}/utils/evaluate.sh ${infer_data} ${ckpt_version}
            fi
            used_ckpt=$last_ckpt
        fi

        if [ "$run_state" == "final" ]; then
            echo "[TRACE] $(date) checkpoint_version is final"  >&2
            break;
        fi
        sleep ${sleep_time}
    done
    return 0
}

function main() {
    config_file="$1"
    config_gpus="$2"
    config_gpu_num="$3"
    mode="$4"
    # VDL_LOG_PATH is set automatically on paddleCloud, only export in local platform
    if [ ! -n "$VDL_LOG_PATH" ];then
        export_visualdl_env
    fi
    train_monitor ${config_file} "${config_gpus}" "${config_gpu_num}" "${mode}"
    return 0
}

main "$@"
