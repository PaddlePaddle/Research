#!/bin/bash 
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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
        ckpt_file="${train_dir_path}/checkpoint.meta"
        if [[ ! -e ${ckpt_file} ]]; then
            echo "[TRACE] $(date) file: ${ckpt_file} does not exist" >&2
            sleep ${sleep_time}
            continue
        fi
        sec_name="Monitor"
        local last_ckpt=$(parse_user_conf ${ckpt_file} "init_pretrain_model" ${sec_name})
        local ckpt_version=$(parse_user_conf ${ckpt_file} "ckpt_version" ${sec_name})
        echo "[TRACE] $(date) last checkpoint is ${last_ckpt}" >&2
        if [ "$used_ckpt" == "$last_ckpt" ]; then
            echo "[TRACE] $(date) ckpt_version: ${ckpt_version}; last_ckpt: ${last_ckpt}; used_ckpt: ${used_ckpt}; sleep"  >&2
        else
            echo "[TRACE] $(date) ckpt_version: ${ckpt_version}; last_ckpt: ${last_ckpt}; used_ckpt: ${used_ckpt}; NEW"  >&2
            infer_data="${train_dir_path}/${ckpt_version}.pred"
            sh -x ${root_dir}/frame/scheduler/run_predict.sh ${config_file} "${config_gpus}" "${config_gpu_num}" "${mode}" ${ckpt_file} 1>${infer_data}
            if [[ $? -eq 0 -a -e ${root_dir}/utils/evaluate.sh ]]; then
                sh -x ${root_dir}/utils/evaluate.sh ${infer_data} ${ckpt_version}
            fi
            used_ckpt=$last_ckpt
        fi

        if [ "$ckpt_version" == "final" ]; then
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
    train_monitor ${config_file} "${config_gpus}" "${config_gpu_num}" "${mode}"
    return $?
}

main "$@"
