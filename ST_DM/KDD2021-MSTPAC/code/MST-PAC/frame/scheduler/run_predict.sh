#!/bin/bash 
root_dir=$(cd $(dirname "$0")/../../; pwd)

source "${root_dir}/frame/scheduler/scheduler_functions.sh"
source "$root_dir/frame/scheduler/hadoop_functions.sh"

function generate_cache_archives() {
    is_update_model="False"
    is_update_code="True"
    if [[ $is_update_model == False ]] && [[ $is_update_code == False ]]; then
        return 0
    fi

    local remote_archives_dir=$1
    cd ${root_dir}
    rm ${root_dir}/paddle-frame.tar.gz 2>/dev/null 
    rm ${root_dir}/model.tar.gz 2>/dev/null 

    if [[ $is_update_code == True ]]; then
        tar zcfh paddle-frame.tar.gz * .env
        if [[ $? -ne 0 ]]; then
            echo "[FATAL] $(date) tar paddle-frame failure." >&2
            return 1
        fi

        hadoop_put_file "${hadoop_home}" "${afs_host}" "${afs_user},${afs_passwd}" \
                        "${root_dir}/paddle-frame.tar.gz" "${remote_archives_dir}"
        if [[ $? -ne 0 ]]; then
            echo "[FATAL] $(date) hadoop put paddle-frame.tar.gz failure." >&2
            return 1
        fi

        echo "[TRACE] $(date) put local paddle-frame to ${remote_archives_dir}" >&2 
    fi

    if [[ $is_update_model == True ]]; then
        local eval_dir=$(parse_user_conf "$config_file" "eval_dir" "Evaluate")
        if [[ ! -d ${eval_dir} ]]; then
            echo "[FATAL] $(date) No eval_dir in ${config_file}" >&2
            return 1
        fi

        cd $eval_dir 
        tar zcf model.tar.gz *
        if [[ $? -ne 0 ]]; then
            echo "[FATAL] $(date) tar model.tar.gz failure." >&2
            return 1
        fi
        mv model.tar.gz ${root_dir}
        cd ${root_dir}
        hadoop_put_file "${hadoop_home}" "${afs_host}" "${afs_user},${afs_passwd}" \
                        "${root_dir}/model.tar.gz" "${remote_archives_dir}"
        if [[ $? -ne 0 ]]; then
            echo "[FATAL] $(date) hadoop put model.tar.gz failure." >&2
            return 1
        fi
        echo "[TRACE] $(date) put local model to ${remote_archives_dir}" >&2 
    fi

    return 0
}

function execute_hadoop_job() {
    local remote_archives_dir=$1
    local job_name="paddle_frame_predict_job_$(date +%s)"
    local mapper_num=2
    local reducer_num=2
    local memory_limit=5000
    local map_output_key_fields=3
    local mapred_separator=";"
    local reducer_separator=";"
    local is_ignoreseparator=true
    local mapper="cat"
    local reducer="./paddle_release_home/python/bin/python ./paddle-frame/frame/core/cpu_predictor.py --conf_file paddle-frame/conf/demo/$(basename $config_file)"
    local input="/user/lbs-mapsearch/wushilei/eval_data/*"
    local output_dir="/user/lbs-mapsearch/wushilei/eval_output"

    ${hadoop_home}/bin/hadoop fs -mv ${output_dir} ${output_dir}.$(date +%s)

    ${hadoop_home}/bin/hadoop streaming \
        -jobconf mapred.job.name=${job_name} \
        -jobconf mapred.job.priority="VERY_HIGH" \
        -jobconf mapred.map.tasks=${mapper_num} \
        -jobconf mapred.reduce.tasks=${reducer_num} \
        -jobconf stream.memory.limit=${memory_limit} \
        -jobconf stream.num.map.output.key.fields=${map_output_key_fields} \
        -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -jobconf stream.reduce.output.field.separator="${reducer_separator}"  \
        -jobconf mapred.textoutputformat.separator="${mapred_separator}" \
        -jobconf mapred.textoutputformat.ignoreseparator=${is_ignoreseparator} \
        -inputformat org.apache.hadoop.mapred.TextInputFormat \
        -outputformat org.apache.hadoop.mapred.TextOutputFormat \
        -cmdenv PYTHONPATH=.:./paddle-frame \
        -cacheArchive "${remote_archives_dir}/paddle-frame.tar.gz#paddle-frame" \
        -cacheArchive "${remote_archives_dir}/model.tar.gz#model" \
        -cacheArchive "${cpu_fluid_afs}#paddle_release_home" \
        -mapper "${mapper}"  \
        -reducer "${reducer}" \
        -input "${input}" \
        -output "${output_dir}" 

    return 0
}

function run_hadoop_scheduler() {
    local remote_archives_dir=/user/lbs-mapsearch/wushilei/
    #cacheArchive files 
    generate_cache_archives ${remote_archives_dir}
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    execute_hadoop_job ${remote_archives_dir}
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    return 0 
}

function run_platform_scheduler() {
    local platform=$1
    local reader=$2
    case ${platform} in 
        local-cpu)
            ${fluid_bin} ${root_dir}/frame/core/cpu_predictor.py --conf_file ${config_file}
            ;;    
        local-gpu)
            if [[ ${core_gpu_num} -gt 1 ]]; then
                ${fluid_bin} -m paddle.distributed.launch --selected_gpus="${select_gpus}" \
                            ${root_dir}/frame/core/gpu_predictor.py --conf_file ${config_file}
            else
                ${fluid_bin} ${root_dir}/frame/core/gpu_predictor.py --conf_file ${config_file} 
            fi
            ;;
        hadoop)
            run_hadoop_scheduler
            ;;
        *)
            ${fluid_bin} ${root_dir}/frame/core/cpu_predictor.py --conf_file ${config_file}
            ;;
    esac

    return 0
}

function main() {
    config_file="$1"
    config_gpus="$2"
    config_gpu_num="$3"
    mode="$4"

    get_platform $1 "Evaluate" "${config_gpu_num}" platform_ref
    echo "[TRACE] $(date) Final platform is ${platform_ref}" >&2

    get_reader $1 "Evaluate" reader_ref
    echo "[TRACE] $(date) Final reader is ${reader_ref}" >&2
    #The interceptor is hook function before run paddle program. 
    #1. Cover the env of conf file with user command line arguments.
    #2. Export env 
    run_interceptor ${platform_ref} "${config_file}" "${config_gpu_num}" "${config_gpus}" "${mode}"
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
