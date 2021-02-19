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

function run_platform_scheduler() {
    local platform=$1
    ret=0
    case ${platform} in 
        local-cpu)
            ${fluid_bin} ${root_dir}/frame/core/cpu_trainer.py --conf_file ${config_file}
            ret=$?
            ;;    
        local-gpu)
            if [[ ${core_gpu_num} -gt 1 ]]; then
                ${fluid_bin} -m paddle.distributed.launch --selected_gpus="${select_gpus}" \
                            ${root_dir}/frame/core/gpu_trainer.py --conf_file ${config_file}
            else
                #nvprof --profile-child-processes \
                ${fluid_bin} ${root_dir}/frame/core/gpu_trainer.py --conf_file ${config_file}
            fi
            ret=$?
            ;;
        pserver-local)
            sh ${root_dir}/frame/scheduler/run_pserver_local.sh ${config_file} 
            ret=$?
            ;;
        *)
            echo "[FATAL] $(date) Invalid platform ${platform}" >&2 
            return 1
    esac

    return $ret
}

function main() {
    config_file="$1"
    config_gpus="$2"
    config_gpu_num="$3"

    get_platform $1 "Train" "${config_gpu_num}" platform_ref
    echo "[TRACE] $(date) Final platform is ${platform_ref}" >&2

    #The interceptor is hook function before run paddle program. 
    #1. Cover the env of conf file with user command line arguments.
    #2. Export env 
    run_interceptor ${platform_ref} "${config_file}" "${config_gpu_num}" "${config_gpus}"
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) run interceptor failure." >&2
        return 1
    fi

    run_platform_scheduler ${platform_ref} 
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) run_platform_scheduler failure." >&2
        return 1
    fi

    return 0
}

main "$@"
