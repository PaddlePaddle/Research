#!/bin/bash

#This utils functions is shared by both trainer and predictor

root_dir=$(dirname $(readlink -f $0))/../..

source "$root_dir/frame/conf/core.conf"
source "$root_dir/conf/var_sys.conf"

#parse user config file
#$1: config_file, such as conf/demo/demo.distributed.conf 
#$2: field_name, such as platform, data_reader
#$3: sec_name, such as Train, Evaluate
function parse_user_conf() {
    local config_file="$1"
    local field_name=$2
    local sec_name=$3
    if [[ ${sec_name} == "" ]]; then
        sec_name="DEFAULT"
    fi

    local field_value=$(${fluid_bin} utils/ini_parser.py \
                            --conf_file ${config_file} --sec_name ${sec_name} --conf_name ${field_name})
    if [[ $field_value == "" ]]; then
        if [[ "$(basename ${config_file})" =~ "distributed" || "$(basename ${config_file})" =~ "paddlecloud" ]]; then
            addition_file=${config_file/distributed/local}
            addition_file=${addition_file/paddlecloud/local}
            addition_file=${addition_file%.tmp} 
            if [[ ! -e ${addition_file} ]]; then
                echo "No local conf file found:"${addition_file} >&2
                exit 1
            fi
            field_value=$(${fluid_bin} utils/ini_parser.py \
                    --conf_file ${addition_file} --sec_name ${sec_name} --conf_name ${field_name})
        fi
    fi

    if [[ ${field_value} == "" ]]; then
        field_kv=$(grep "^${field_name}=" ${root_dir}/.env)
        field_value=${field_kv#*=}
    fi

    echo "${field_value}"
}

#get platform config in config_file
#$1: config_file, such as conf/demo/demo.distributed.conf 
#$2: sec_name, such as Train, Evaluate
#$3: config_gpu_num, user args
#$4: param_platform, eval
function get_platform() {
    local config_file=$1
    local sec_name=$2
    local config_gpu_num=$3 
    local param_platform=$4
    local platform=$(parse_user_conf ${config_file} "platform" ${sec_name})
    if [[ $platform == "" ]]; then
        platform="local-cpu"
        echo "[Warning] $(date) Platform is null. Set cpu as default." >&2
    fi
    
    local is_invalid_platform=$(echo "${core_all_platform}" | egrep -o " ${platform} ")
    if [[ $is_invalid_platform == "" ]]; then
        echo "[Warning] $(date) Platform ${platform} is not in \
             ${core_all_platform}. Set cpu as default." >&2
        platform="local-cpu"      
    fi

    if [[ ${config_gpu_num} != "" ]]; then
        if [[ ${config_gpu_num} == "0" ]]; then
            platform="local-cpu"
        fi

        if [[ $((config_gpu_num)) -gt 0 ]]; then 
            platform="local-gpu"
        fi
    fi

    eval $param_platform=$platform

    return 
}

#get data_reader form config_file
#$1: config_file, such as conf/demo/demo.distributed.conf 
#$2: sec_name, such as Train, Evaluate
#$3: param_reader, eval value
function get_reader() {
    local config_file=$1
    local sec_name=$2
    local param_reader=$3
    local reader=$(parse_user_conf ${config_file} "data_reader" ${sec_name})

    if [[ $reader == "" ]]; then
        reader="dataset"
        echo "[Warning] $(date) data_reader is null. set data_reader:dataset as default." >&2
    fi

    local is_invalid_reader=$(echo "${core_all_reader}" |egrep -o " ${reader} ")
    if [[ $is_invalid_reader == "" ]]; then
        echo "[Warning] $(date) Reader: ${reader} is not in \
             ${core_all_reader}. set dataset as default." >&2
        reader="dataset"
    fi

    eval $param_reader=$reader
    return 
}

#calculate env: input is core.conf pattern, replace variable with its values
#$1: input values in core.conf
#$2: config_file, such as conf/demo/demo.distributed.conf 
#$3: sec_name, such as Train, Evaluate
function calculate_env_val() {
    local values=$1
    local config_file=$2
    local sec_name=$3
    old_ifs=$IFS
    IFS=$'+'
    ret=""
    for conf_val in $values; do
        # "$" indicates variable, no "$" indicates const_string
        if [[ -z $(echo "$conf_val" | grep '\$') ]]; then
            ret=${ret}${conf_val}
            continue
        fi
        conf_val=${conf_val#*$}

        # get value from current conf
        local cur_value=$(parse_user_conf ${config_file} ${conf_val} ${sec_name})
        if [[ $cur_value == "" ]]; then
            #ret=${ret}${conf_val}
            continue
        else
            ret=${ret}${cur_value}
        fi
    done 
    IFS=$old_ifs
    echo "$ret"
}

function dump_env_var() {
    local kv="$1"
    local env_file=$2

    local key=${kv%=*}
    local is_exist=$(grep ^${key} ${env_file})
    if [[ -z $is_exist ]]; then
        echo "${kv}" >> ${env_file}
    else
        sed -i /^${key}/c${kv} ${env_file}
    fi
}

#set env for frame
#$1: config_file, such as conf/demo/demo.distributed.conf 
#$2: evn_list, env variables in current function call
#$3: is_clear, rm previous env or not
function set_env() {
    local config_file="${1}"
    local env_list="${2}"
    local is_clear="${3}"
   
    local env_file="${root_dir}/.env"
    if [[ $is_clear == "1" ]]; then
        > "${env_file}" 
    fi

    error="0"
    for env_var in $env_list; do
        if [[ $env_var =~ "=" ]]; then
            source ${env_file}
            key=${env_var%=*}
            values=${env_var#*=}
            final_value=$(calculate_env_val ${values} ${config_file})
            #echo "${key}=${final_value}" >> ${env_file}
            if [[ -z $final_value ]]; then
                error="1"
                continue
            fi
            dump_env_var "${key}=${final_value}" ${env_file}
            continue
        fi
        new_env_var=${env_var#*:}
        env_var=${env_var%:*} 
        cur_env=$(grep "${env_var}" ${config_file} |grep -v "#" | tail -1 |sed 's|:|=|;s| ||g')
        
        if [[ $cur_env == "" ]]; then
            if [[ "$(basename ${config_file})" =~ "distributed" || "$(basename ${config_file})" =~ "paddlecloud" ]]; then
                addition_file=${config_file/distributed/local}
                addition_file=${addition_file/paddlecloud/local}
                addition_file=${addition_file%.tmp} 
                if [[ ! -e ${addition_file} ]]; then
                    echo "No local conf file found:"${addition_file} >&2
                    exit 1
                fi

                cur_env=$(grep "${env_var}" ${addition_file} | grep -v "#" | \
                         tail -1 |sed 's|:|=|;s| ||g')
            fi
        fi

        if [[ $cur_env == "" ]]; then
            is_optional=$(echo "${option_env_list}" | egrep -o " ${env_var} ")
            if [[ ${is_optional} != "" ]]; then
                continue
            fi
            echo "[Warning] $(date) Please check [${env_var}] config in ${config_file}" >&2
            #continue and mark at error
            error="1"
        else
            #echo ${cur_env} >> "${env_file}"
            if [[ ${env_var} != ${new_env_var} ]]; then
                #echo "${new_env_var}=${cur_env#*=}" >> ${env_file}
                dump_env_var "${new_env_var}=${cur_env#*=}" ${env_file}
            else
                dump_env_var ${cur_env} ${env_file}
            fi
        fi
    done
    echo "conf_file=$config_file" >> ${env_file}
    source ${env_file}
    #set permission for all users
    chmod +rwx ${env_file}

    if [[ $error == "1" ]]; then
        return 1
    fi

    return 0
}

#export visual_dl log path
function export_visualdl_env() {
    vdl_dir_path=$(parse_user_conf $config_file "visualdl_dir" "DEFAULT")
    export VDL_LOG_PATH=${vdl_dir_path}
    echo "[DEBUG] $(date) VDL_LOG_PATH is $VDL_LOG_PATH" >&2
    return
}

#export cuda env, such as LD_LIBRARY_PATH
function export_cuda_env() {
    local config_gpus=$1
    if [[ ${cuda_lib_path} != "" ]]; then
        export LD_LIBRARY_PATH=${cuda_lib_path}:$LD_LIBRARY_PATH
    fi

    if [[ ${config_gpus} != "" ]]; then
        export CUDA_VISIBLE_DEVICES=${config_gpus}
    elif [[ ${CUDA_VISIBLE_DEVICES} != "" ]]; then
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    fi

    echo "[DEBUG] $(date) LD_LIBRARY_PATH is $LD_LIBRARY_PATH" >&2
    return
}

#set gpu num for frame. gpu num may come from user args, user conf, or compution from other config
function set_core_gpu_num() {
    local config_gpu_num=$1
    local config_gpus=$2
    if [[ ${config_gpu_num} != "" ]]; then
        echo "core_gpu_num=${config_gpu_num}" >> "${root_dir}/.env"
    elif [[ ${config_gpus} != "" ]]; then
        core_gpu_num=$(echo ${config_gpus} | sed "s/,/\n/g"|wc -l)
        echo  "core_gpu_num=${core_gpu_num}" >> "${root_dir}/.env"
    elif [[ ${CUDA_VISIBLE_DEVICES} != "" ]]; then
        core_gpu_num=$(echo ${CUDA_VISIBLE_DEVICES} | sed "s/,/\n/g"|wc -l)
        echo  "core_gpu_num=${core_gpu_num}" >> "${root_dir}/.env"
    else
        echo "core_gpu_num=1" >> "${root_dir}/.env" 
    fi
    source "${root_dir}/.env"

    return 0
}

#get select gpus for multi-gpu platform
function get_select_gpus() {
    local gpu_num=$1
    local visible_gpu_index=${CUDA_VISIBLE_DEVICES}

    local visible_num=$(echo "${visible_gpu_index}" | sed "s/,/\n/g"| wc -l)
    select_gpus=""
    if [[ $visible_num -lt ${gpu_num} ]]; then
        gpu_num=${visible_num}
    fi
    if [[ $visible_num -ge $gpu_num ]]; then
        for ((i = 0; i < $gpu_num; i++)) do {
            select_gpus=${select_gpus}",${i}"
        } 
        done
        select_gpus=${select_gpus#,*}
        select_gpus=$visible_gpu_index
    fi

    echo "${select_gpus}"
}

function get_conf_gpu_num() {
    config_file=$1
    local gpu_per_node_num=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name "k8s_gpu_cards|gpu_per_node_num")
    if [[ -z ${gpu_per_node_num} ]]; then
        gpu_per_node_num=1
    fi
    local workers_per_node_num=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name workers_per_node_num)
    if [[ -z $workers_per_node_num ]]; then
        workers_per_node_num=1
    fi
    local config_gpu_num=$((${gpu_per_node_num} / ${workers_per_node_num}))
    echo ${config_gpu_num}
}

#generate env file for frame.
#$1: platform, such as pserver-cpu, local-cpu, slurm, hadoop
#$2: config_file, such as conf/demo/demo.distributed.conf 
#$3: config_gpu_num, user args
#$4: config_gpus, user args
function env_interceptor() {
    local platform=$1
    local config_file=$2
    local config_gpu_num=$3
    local config_gpus=$4
    > ${root_dir}/.env

    if [[ ${platform} == "local-gpu" ]]; then
        #ignore return for set_env. It's unnecessary here
        set_env "${config_file}" "${cuda_env_list}" "1"
        set_core_gpu_num "${config_gpu_num}" "${config_gpus}"
        export_cuda_env "${config_gpus}"
        select_gpus=$(get_select_gpus "${core_gpu_num}")
    elif [[ ${platform} == "pserver-cpu" || ${platform} == "pserver-gpu" ]]; then
        paddlecloud_env_list=$(echo "${paddlecloud_env_list}" | tr -d "\n" | sed "s/[ ][ ]*/ /g")
        echo "paddlecloud_env_list ${paddlecloud_env_list}" >&2
        set_env "${config_file}" "${paddlecloud_env_list}" "1"
        echo "thirdparty_path=${thirdparty_path}" >> ${root_dir}/.env
        echo "paddlecould_fluid_bin=${paddlecould_fluid_bin}" >> ${root_dir}/.env
        if [ "${platform}" == "pserver-gpu" ];then 
            config_gpu_num=$(get_conf_gpu_num ${config_file})
            set_core_gpu_num "${config_gpu_num}" "${config_gpus}"
        fi
        generate_config_ini ${config_file}
    elif [[ ${platform} == "slurm" ]]; then  
        slurm_env_list=$(echo "${slurm_env_list}" | tr -d "\n" | sed "s/[ ][ ]*/ /g")
        set_env "${config_file}" "${slurm_env_list}" "1"
        config_gpu_num=$(get_conf_gpu_num ${config_file})
        set_core_gpu_num "${config_gpu_num}" "${config_gpus}"
        set_train_dir ${config_file}
    elif [[ ${platform} == "hadoop" ]]; then
        paddlecloud_env_list=$(echo "${paddlecloud_env_list}" | tr -d "\n" | sed "s/[ ][ ]*/ /g")
        set_env "${config_file}" "${paddlecloud_env_list}" "1"
    fi
    
    if [[ ${platform} == "pserver-cpu" || ${platform} == "hadoop" ]]; then 
        echo "fluid_afs=${cpu_fluid_afs}" >> ${root_dir}/.env
    elif [[ ${platform} == "pserver-gpu" || ${platform} == "slurm" ]]; then
        echo "fluid_afs=${gpu_fluid_afs}" >> ${root_dir}/.env
    fi

    return 0
}

#modify user conf for frame. It should be roll back in postprocess.
#$1: platform, such as pserver-cpu, local-cpu, slurm, hadoop
#$2: config_file, such as conf/demo/demo.distributed.conf 
#$3: config_gpu_num, user args
function user_conf_interceptor() {
    local platform=$1
    local config_file=$2
    local core_gpu_num=$3
    local mode=$4
    
    echo -e "\n[USERARGS]\n" >> $config_file
    if [[ $core_gpu_num -ge 1 ]]; then
        echo "num_gpus: ${core_gpu_num}" >> ${config_file}
    fi
    dataset_name=$(parse_user_conf $config_file dataset_name)
    model_name=$(parse_user_conf $config_file model_name)

    
    mount_local_data_dir=./${mount_local_dir}

    if [[ $platform == "slurm" ]]; then
        if [[ ${dataset_name} == "" ]]; then 
            echo "[FATAL] $(date) dataset_name is null." >&2
            return 1
        fi
        (
		cat <<- SLURMEOF
fluid_bin: ./tools/paddle_release_home/python/bin/python
dataset_dir: ${mount_local_data_dir}
train_dir: ./output
		SLURMEOF
        ) >> ${config_file}

    elif [[ ${platform} == "pserver-cpu" || ${platform} == "pserver-gpu" ]]; then
        echo "fluid_bin: python" >> ${config_file}
        if [ "$data_location" == "local" ];then 
            echo "dataset_dir: ./train_data" >> ${config_file}
        else
            echo "dataset_dir: ${mount_local_data_dir}" >> ${config_file}
        fi
        echo "train_dir: ./output" >> ${config_file}
    elif [[ ${platform} == "hadoop" ]]; then
        echo "fluid_bin: ./paddle_release_home/python/bin/python" >> ${config_file}
        echo "eval_dir: ./model" >> ${config_file}
    else
        echo "fluid_bin: ${fluid_bin}" >> ${config_file} 
    fi

    import_modules=""
    for user_module in ${user_defined_modules}; do
        sub_dir=${user_module%:*}
        base_class=${user_module#*:}

        cur_file_list=$(find -L ${root_dir}/${sub_dir} -name "*.py" -printf '%P\n')
        for sub_module in $cur_file_list; do
            class=$(cat ${root_dir}/${sub_dir}/${sub_module} | tr -d " \n" | egrep -o "class[a-zA-Z0-9]+\(${base_class}\)")
            if [[ $class == "" || ! ("$class" =~ "$dataset_name" || "$class" =~ "$model_name") ]]; then
                continue
            fi 
            module_name=${sub_module/\//.}
            import_modules=${import_modules}",${sub_dir}.${module_name%.*}" 
        done
    done
    echo "import_user_modules: ${import_modules#,*}" >> ${config_file}

    if [[ $mode == "monitor" ]]; then
        local train_dir_path=$(parse_user_conf $config_file "train_dir" "Train")
        local ckpt_file="${train_dir_path}/checkpoint.meta"
        if [[ -e ${ckpt_file} ]]; then
            echo "[TRACE] $(date) ckpt_file is ${ckpt_file}" >&2
            # add init_pretrain_model and eval_dir to config_file, the last section of config file is evaluate section
            sed -n '2,$p' ${ckpt_file} >> ${config_file}
        fi
    fi
    return 0
}

#set train_dir for slurm afs_mount.
#$1: config_file, such as conf/demo/demo.distributed.conf 
#$2: env_file, the frame env file.
function set_train_dir() {
    local config_file=$1
    local env_file="${root_dir}/.env"
    model_name=$(parse_user_conf $config_file model_name)
    if [[ ${model_name} == "" ]]; then
        echo "[FATAL] $(date) model_name is null." >&2
        return 1
    fi
    echo "slurm_local_train_dir=./${mount_local_dir}/model/${model_name}/${job_name}" >> ${env_file}
    return 0
}

#generate_config_ini for paddlecloud. 
function generate_config_ini() {
    local config_file=$1
    cd ${root_dir}
    paddle_cloud_conf_ini="$(dirname ${config_file})/config.ini"
    cp ${root_dir}/.env ${paddle_cloud_conf_ini}
    return 0
}

#generate env an modify user config file for the whole frame.
function run_interceptor() {
    local platform=$1
    local config_file=$2
    local config_gpu_num=$3
    local config_gpus=$4
    local mode=$5
    env_interceptor "${platform}" "${config_file}" "${config_gpu_num}" "${config_gpus}"
    if [[ $? != 0 ]]; then
        return 1
    fi
    
    source ${root_dir}/.env
    user_conf_interceptor "${platform}" "${config_file}" "${core_gpu_num}" "${mode}"
    export_visualdl_env
    if [[ $? != 0 ]]; then
        return 1
    fi

    return 0
}

