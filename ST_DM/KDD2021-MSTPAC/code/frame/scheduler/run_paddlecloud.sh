#!/bin/bash 

root_dir=$(cd $(dirname "$0")/../../; pwd)

function prepare_code_uri() {
    cd ${root_dir}
    rm ${root_dir}/paddle-frame.tar.gz 2>/dev/null 
    cp ${root_dir}/frame/scheduler/paddlecloud/before_hook.sh before_hook.sh
    cp ${root_dir}/frame/scheduler/paddlecloud/end_hook.sh end_hook.sh
    cp ${root_dir}/.env ${root_dir}/env_paddlecloud
    tar zcfh paddle-frame.tar.gz * .env
    rm ${root_dir}/env_paddlecloud
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) tar paddle-frame.tar.gz failure." >&2
        return 1
    fi
    rm before_hook.sh end_hook.sh 2>/dev/null
    ${hadoop_home}/bin/hadoop fs -mv ${code_uri} ${code_uri}.$(date +%s)
    if [[ $? -ne 0 ]]; then
        echo "[Warning] $(date) backup ${code_uri} failure. Ignore and continue.." >&2
    fi

    hadoop_put_file "${hadoop_home}" "${fs_name}" "${fs_ugi}" \
                        "${root_dir}/paddle-frame.tar.gz" "${code_uri}"
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) hadoop put paddle-frame.tar.gz failure." >&2
        return 1
    fi

    return 0
}

function init_job() {
    source ${root_dir}/conf/var_sys.conf
    source ${root_dir}/.env 
    source ${root_dir}/frame/scheduler/hadoop_functions.sh 
    return 0
}

function prepare_output() {

    ${hadoop_home}/bin/hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} \
                                -test -e ${fs_name}${output_path}
    if [[ $? -eq 0 ]]; then
        return 0
    fi

    ${hadoop_home}/bin/hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} \
                                -mkdir ${fs_name}${output_path} 
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) mkdir ${fs_name}${output_path} failure." >&2
        return 1
    fi

    return 0
}

function prepare_config_ini() {
    conf_relative_name=$(echo $config_file | awk -F"paddle-frame/" '{print $NF}')
        
    job_conf="$(dirname ${config_file})/config.ini"
    echo "conf_relative_name ... ${conf_relative_name}" >&2 
    #echo "GLOG_v=4" >> $job_conf
    #echo "GLOG_logtostderr=1" >> $job_conf
    if [ "${data_location}" == "local" ];then
        echo "train_data_path=$data_path" >> $job_conf
    else
        echo "mount_afs=\"true\"" >> $job_conf
    fi

    return 0
}

function prepare_group_name() {
    if [[ -n $group_name ]] ||
       [[ ${mode} != "gpu" ]] || 
       [[ ${paddlecloud_group_type} != "k8s" ]]; then
        return 0
    fi

    if [[ -z ${pdc_condidates_groupname} ]]; then
        return 1
    fi
    if [[ -z ${k8s_gpu_cards} ]]; then
        k8s_gpu_cards=4
    fi
    if [[ -z ${k8s_trainers} ]]; then
        k8s_trainers=1
    fi

    local default_groupname=""
    local user_cluster_info=$(paddlecloud cluster list)

    for candi_groupname in ${pdc_condidates_groupname}; do
        has_permision=$(echo ${user_cluster_info} | grep ${candi_groupname})
        if [[ -z ${has_permision} ]]; then
            continue
        fi

        default_groupname=${candi_groupname}
        paddle_group_info=$(paddlecloud group info --group-name ${candi_groupname} --json)
        idle_gpu_num=$(echo ${paddle_group_info} | tr ',{' '\n' | sed -n '/idleQuota/,/}/p'  |grep "gpu':" | egrep -o [0-9]*)

        if [[ -z ${idle_gpu_num} ]] || [[ ${idle_gpu_num} -lt ${k8s_gpu_cards} ]]; then
            continue
        fi

        group_name=${candi_groupname}
        break
    done 

    if [[ -z ${default_groupname} ]]; then
        return 1
    fi

    if [[ -z ${group_name} ]]; then
        group_name=${default_groupname}
    fi

    sed -i "/^num_gpus/cnum_gpus: ${k8s_gpu_cards}" ${config_file}
    return 0
}

function prepare_job() {
    prepare_group_name
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) Please set group_name in ${config_file}" >&2
        return 1
    fi


    prepare_code_uri
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) generate code_uri failure." >&2
        return 1
    fi

    prepare_output
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) prepare output failure." >&2
        return 1
    fi
    
    prepare_config_ini
    
    return 0
}

function append_extra_args() {
    local key=$1
    local value=$2
    local is_pre=0
    if [[ -n $3 ]]; then
        is_pre=$3
    fi

    if [[ $key == "" ]] || [[ $value == "" ]]; then
        return
    fi
    if [[ $is_pre -ne 0 ]]; then
        pre_extra_args="${pre_extra_args} --${key} ${value}"
    else
        extra_args="${extra_args} --${key} ${value}"
    fi
    return 0
}

function train_job() {
    #padlecloud bin dir is set at PATH
    pre_extra_args=""
    extra_args=""
    append_extra_args ak "${ak}" 1
    append_extra_args sk "${sk}" 1

    if [ "$mode" == "cpu" ] && [ ! -n "${tensorflow_version}" ];then
        paddlecloud job ${pre_extra_args} train --job-name ${job_name} \
                        --group-name ${group_name} \
                        --job-conf ${job_conf} \
                        --code-uri "${code_uri}" \
                        --start-cmd "sh ./frame/scheduler/paddlecloud/run_cmd.sh ${conf_relative_name} cpu ${need_evaluate} ${thirdparty_paddle_version}" \
                        --job-version ${paddle_version} \
                        --k8s-trainers ${k8s_trainers} \
                        --k8s-ps-num ${k8s_ps_num} \
                        --k8s-cpu-cores ${k8s_cpu_cores} \
                        --k8s-memory ${k8s_memory} \
                        --k8s-ps-cores ${k8s_ps_cores} \
                        --k8s-ps-memory ${k8s_ps_memory} \
                        --k8s-priority ${k8s_priority} \
                        --is-auto-over-sell ${is_auto_over_sell} \
                        --wall-time ${wall_time} \
                        --distribute-job-type "PSERVER" \
                        --is-standalone 0 ${extra_args} 
    elif [ "$mode" == "gpu" ] && [ ! -n "${tensorflow_version}" ];then
        if [ "$paddlecloud_group_type" == "slurm" ];then
            if [ ${slurm_nodes} -eq 1 ];then
                append_extra_args "is-standalone" 1
            else
                append_extra_args "distribute-job-type" "NCCL2"
            fi
            #paddlecloud test env
            #paddlecloud job --ak 201baf60e552517aa313e3d5b94260ed --sk 9de38f4ff04556e3a45e87f12286158c \
            #               --server paddlecloud-integration.baidu-int.com --port 8800 train --job-name ${job_name} \
            paddlecloud job ${pre_extra_args} train --job-name ${job_name} \
                            --group-name ${group_name} \
                            --job-conf ${job_conf} \
                            --code-uri "${code_uri}" \
                            --start-cmd "sh ./frame/scheduler/paddlecloud/run_cmd.sh ${conf_relative_name} gpu ${need_evaluate} ${thirdparty_paddle_version}" \
                            --job-version ${paddle_version} \
                            --slurm-nodes ${slurm_nodes} \
                            --slurm-gpu-pnode ${slurm_gpu_pnode} \
                            --slurm-task-pnode ${slurm_task_pnode} \
                            --wall-time ${wall_time} ${extra_args} 
        elif [ "$paddlecloud_group_type" == "k8s" ];then
            if [ "${data_location}" != "local" ];then
                echo "afs_local_mount_point=\"/root/paddlejob/workspace/env_run/afs/\"" >> $job_conf
            fi
            if [ ${k8s_trainers} -eq 1 ];then
                append_extra_args "is-standalone" 1
            else
                append_extra_args "distribute-job-type" "NCCL2"
            fi
            #paddlecloud test env
            #paddlecloud job --ak 201baf60e552517aa313e3d5b94260ed --sk 9de38f4ff04556e3a45e87f12286158c \
            #               --server paddlecloud-integration.baidu-int.com --port 8800 train --job-name ${job_name} \
            paddlecloud job ${pre_extra_args} train --job-name ${job_name} \
                            --group-name ${group_name} \
                            --job-conf ${job_conf} \
                            --code-uri "${code_uri}" \
                            --start-cmd "sh ./frame/scheduler/paddlecloud/run_cmd.sh ${conf_relative_name} gpu ${need_evaluate} ${thirdparty_paddle_version}" \
                            --job-version ${paddle_version} \
                            --k8s-trainers ${k8s_trainers} \
                            --k8s-gpu-cards ${k8s_gpu_cards} \
                            --k8s-priority ${k8s_priority} \
                            --is-auto-over-sell ${is_auto_over_sell} \
                            --wall-time ${wall_time} ${extra_args} 
        fi
    else
        #tensorflow
        echo "storage_type=\"afs\"" >> $job_conf
        echo "force_reuse_output_path=\"True\"" >> $job_conf
        echo "afs_local_mount_point=\"/root/paddlejob/workspace/env_run/afs/\"" >> $job_conf
        if [ ${k8s_ps_num} -eq 1 ] && [ ${k8s_trainers} -eq 1 ];then
            append_extra_args "is-standalone" 1
        elif [ ${k8s_ps_num} -ne 1 ];then
            k8s_trainers="1"
            append_extra_args "distribute-job-type" "PSERVER"
            append_extra_args "k8s-ps-num" "${k8s_ps_num}"
        elif [ ${k8s_trainers} -ne 1 ];then
            k8s_ps_num="1"
            append_extra_args "distribute-job-type" "NCCL2"
        fi
        paddlecloud job ${pre_extra_args} --debug train --job-name ${job_name} \
        --group-name ${group_name} \
        --job-conf ${job_conf} \
        --start-cmd "sh ./frame/scheduler/paddlecloud/run_tf.sh" \
        --code-uri "${code_uri}" \
        --job-version ${tensorflow_version}  \
        --k8s-trainers ${k8s_trainers} \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --is-auto-over-sell ${is_auto_over_sell} \
        --wall-time ${wall_time} ${extra_args} 
    fi

    return 0
}

function main() {
    config_file="$1"
    mode=$2
    init_job

    prepare_job
    if [[ $? -ne 0 ]]; then
        echo "[FATAL] $(date) prepare paddlecloud job env failulre" >&2
        return 1
    fi

    train_job
    return 0
}

main "$@"
