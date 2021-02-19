#!/bin/sh

CUR_PATH=$(echo ${BASH_SOURCE} | xargs dirname)
ROOT_PATH=$(readlink -f $CUR_PATH/../../)

LOCAL_CONF=${ROOT_PATH}/conf/nmt/nmt.local.conf
DISTRIBUTED_CONF=${ROOT_PATH}/conf/nmt/nmt.distributed.conf

function set_env() {
    afs_user=$(grep ^afs_user ${DISTRIBUTED_CONF} | sed 's/afs_user://g;s/ //g')
    afs_passwd=$(grep ^afs_passwd ${DISTRIBUTED_CONF} | sed 's/afs_passwd://g;s/ //g')
    afs_mount_path=$(grep ^afs_mount_path ${DISTRIBUTED_CONF} | sed 's/afs_mount_path://g;s/ //g')
    
    hadoop fs -Dhadoop.job.ugi=${afs_user},${afs_passwd} -rmr ${afs_mount_path}/model/NMT/${job_name}
    hadoop fs -Dhadoop.job.ugi=${afs_user},${afs_passwd} -mkdir ${afs_mount_path}/model/NMT/${job_name}
    hadoop fs -Dhadoop.job.ugi=${afs_user},${afs_passwd} -mkdir ${afs_mount_path}/libs/NMT${queue_name}
     
    #set priority
    sed -i "/^k8s_priority=/ck8s_priority=\"low\"" ${ROOT_PATH}/conf/var_sys.conf
    sed -i "s/code_uri.*/code_uri=\$afs_mount_path+libs\/+\$dataset_name+${queue_name}+\/paddle-frame.tar.gz/g" ${ROOT_PATH}/frame/conf/core.conf
    #set vocab
    cp ${ROOT_PATH}/../tmp/data/nmt/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 ${ROOT_PATH}/test/
    
    sed -i "/^src_vocab_fpath:/csrc_vocab_fpath: ./test/vocab_all.bpe.32000" ${LOCAL_CONF}
    sed -i "/^trg_vocab_fpath:/ctrg_vocab_fpath: ./test/vocab_all.bpe.32000" ${LOCAL_CONF}
    
    sed -i "s/^job_name:.*/job_name: nmt_job_${queue_name}/g" ${DISTRIBUTED_CONF}
    sed -i "s/^gpu_per_node_num:.*/gpu_per_node_num: ${gpu_num}/g" ${DISTRIBUTED_CONF}
    sed -i "s/^queue_name:.*/queue_name: ${queue_name}/g" ${DISTRIBUTED_CONF}
    sed -i "s/^max_number_of_steps:.*/max_number_of_steps: 80000/g" ${LOCAL_CONF}
    
    return 0
}

function parse_args() {
    queue_name=$1
    gpu_num=$2
    job_name="nmt_job_${queue_name}"

    paddle_job_len=$(paddlecloud job list | wc -l)
    if [[ $paddle_job_len -lt 200 ]]; then
        echo "[FATAL] $(date) paddlecloud job list len ${paddle_job_len} less than 200."
        return 1
    fi

    is_running=$(paddlecloud job list --size 60 | egrep "groupName|Status" | \
            sed '/Status/N;s/\n//g;s/ //g' | grep ${queue_name} | \
            egrep "submit|schedule|queue|running")
    if [[ $is_running != "" ]]; then
        echo "[NOTICE] $(date) ${queue_name} is running."
        return 1
    fi
    running_info=$(paddlecloud job list --size 60 | egrep "groupName|Status" | \
            sed '/Status/N;s/\n//g;s/ //g' | grep ${queue_name})
    echo "cur_info is ${running_info}"
    
    return 0
}

function run_job() {
    cd ${ROOT_PATH}
    sh run.sh -c nmt.distributed
    
    return 0
}

function main() {
    parse_args "$@"
    if [[ $? -ne 0 ]]; then
        return 1
    fi
        
    set_env
    run_job
    return 0
}

main "$@"
