#!/bin/bash 
root_dir=$(cd $(dirname "$0")/../../; pwd)
source "${root_dir}/frame/scheduler/scheduler_functions.sh"

function run_platform_scheduler() {
    local platform=$1
    local reader=$2
    case ${platform} in 
        local-cpu)
            ${fluid_bin} ${root_dir}/frame/core/cpu_trainer.py --conf_file ${config_file}
            ;;    
        local-gpu)
            if [[ ${core_gpu_num} -gt 1 ]]; then
                ${fluid_bin} -m paddle.distributed.launch --selected_gpus="${select_gpus}" \
                            ${root_dir}/frame/core/gpu_trainer.py --conf_file ${config_file}
            else
                #nvprof --profile-child-processes \
                ${fluid_bin} ${root_dir}/frame/core/gpu_trainer.py --conf_file ${config_file}
            fi
            ;;
        pserver-local)
            sh ${root_dir}/frame/scheduler/run_pserver_local.sh ${config_file} 
            ;;
        pserver-cpu)
            sh ${root_dir}/frame/scheduler/run_paddlecloud.sh ${config_file} "cpu" 
            ;;
        pserver-gpu)
            sh -x ${root_dir}/frame/scheduler/run_paddlecloud.sh ${config_file} "gpu"
            ;;
        slurm)
            sh ${root_dir}/frame/scheduler/run_slurm.sh ${config_file}
            ;;
        *)
            echo "[FATAL] $(date) Invalid platform ${platform}" >&2 
            return 1
    esac

    return 0
}

function main() {
    config_file="$1"
    config_gpus="$2"
    config_gpu_num="$3"

    get_platform $1 "Train" "${config_gpu_num}" platform_ref
    echo "[TRACE] $(date) Final platform is ${platform_ref}" >&2

    get_reader $1 "Train" reader_ref
    echo "[TRACE] $(date) Final reader is ${reader_ref}" >&2

    #The interceptor is hook function before run paddle program. 
    #1. Cover the env of conf file with user command line arguments.
    #2. Export env 
    run_interceptor ${platform_ref} "${config_file}" "${config_gpu_num}" "${config_gpus}"
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) run interceptor failure." >&2
        return 1
    fi

    run_platform_scheduler ${platform_ref} ${reader_ref} 
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) run_platform_scheduler failure." >&2
        return 1
    fi

    return 0
}

main "$@"
