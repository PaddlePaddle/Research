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

root_dir=$(dirname $(readlink -f $0))

source "$root_dir/frame/conf/core.conf"
export PYTHONPATH=${PYTHONPATH}:${root_dir}
#export GLOG_v=5
#export GLOG_logtostderr=1
#export FLAGS_fraction_of_gpu_memory_to_use=1.0

function set_default() {
    config_mode="train"
    config_file="./conf/xxx.local.conf"
    config_process_num="1"
    config_gpus=""
    config_gpu_num=""
    config_root_dir="$root_dir"

    return 0
}

function help_message() {
    echo "sh run.sh -c conf/xxx.local.conf -m train" >&2
    return 0
}

function parse_args() {
    while true; do
        if [[ $1 == "" ]]; then
            return 0
        fi

        case $1 in 
            -c | --conf) config_file=$2;
                         shift 2;;
            -m | --mode) config_mode=$2;
                         shift 2;;
            -g | --gpus) config_gpus=$2;
                         shift 2;;
            -n | --gpu_num) config_gpu_num=$2;
                        shift 2;;
            -h | --help) help_message; shift; exit 0;;
            *) echo "[FATAL] $(date) Invalid arguments!" >&2; exit 1;;
        esac
    done

}

function catch_exception() {
    local dir_mode=0
    if [[ ! -e ${config_root_dir}/${config_file} ]]; then
        #not absolute path, try to use relative path
        dir_mode=1
    else
        config_file=${config_root_dir}/${config_file}
    fi

    if [[ $dir_mode -eq 1 ]] && [[ ! -e ${config_file} ]]; then
        #both absolute and relative path is not valid. try to use reg match
        dir_mode=2
    fi

    if [[ $dir_mode -eq 2 ]]; then
        local conf_num=$(find -L ${config_root_dir} -name "*${config_file}*" | wc -l)
        if [[ $conf_num -lt 1 ]]; then
            echo "[FATAL] $(date) Conf file is not exist. ${config_file}" >&2
            return 1
        fi
        
        local final_config_file=$(find -L ${config_root_dir} -name "*${config_file}*" \
                | grep "conf$"| tail -1)
        if [[ ! -e ${final_config_file} ]]; then
            echo "[FATAL] $(date) Conf file is not exist. ${config_file}" >&2
            return 1
        else
            echo "[TRACE] $(date) Instead conf file as ${final_config_file}" >&2
            config_file=${final_config_file}
        fi
    fi 
    local is_invalid_mode=$(echo "${core_all_mode}" | egrep -o " ${config_mode} ")
    if [[ ${is_invalid_mode} == "" ]]; then
        echo "[FATAL] $(date) Invalid mode ${config_mode}" >&2
        return 1
    fi

    return 0
}

function init_processor() {
    #init conf object
    set_default

    #parse arguments
    parse_args "$@"

    #check args
    catch_exception
    if [ $? -ne 0 ]; then
        return 1
    fi
    #backup conf
    cp ${config_file} ${config_file}.tmp
    config_file=${config_file}.tmp
    return 0
}

function end_processor() {
    mkdir -p ${config_root_dir}/logs/
    mv ${config_root_dir}/*prototxt ${config_root_dir}/logs/ 2>/dev/null

    #delete user defined params
    cp ${config_file} /tmp/user_defined.conf 2>/dev/null
    mv $(dirname ${config_file})/config.ini /tmp/config.ini 2>/dev/null
    return 0
}

function run_mode_scheduler() {
    local schedule_abs_dir="${config_root_dir}/frame/scheduler"

    ret=0
    case $config_mode in 
        train) 
            sh -x ${schedule_abs_dir}/run_train.sh ${config_file} "${config_gpus}" "${config_gpu_num}"
            ret=$?
            ;;
        predict)
            sh -x ${schedule_abs_dir}/run_predict.sh ${config_file} "${config_gpus}" "${config_gpu_num}" "${config_mode}"
            ret=$?
            ;;
        monitor)
            sh -x ${schedule_abs_dir}/run_monitor.sh ${config_file} "${config_gpus}" "${config_gpu_num}" "${config_mode}"
            ret=$?
            ;;
        *)
            echo "[FATAL] $(date) mode is invalid: ${config_mode}" >&2
            return 1
    esac
    return $ret
}

function main() {
    echo "[Trace] $(date) Start running program.." >&2
    #init 
    init_processor "$@"
    if [ $? -ne 0 ]; then
        echo "[FATAL] $(date) Init processor failure." >&2
        return 1
    fi

    run_mode_scheduler
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Run scheduler failure." >&2
        return 1
    fi

    end_processor
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) End processor failure." >&2
        return 1
    fi
    echo "[Trace] $(date) End of running program.." >&2
    return 0
}

main "$@"
