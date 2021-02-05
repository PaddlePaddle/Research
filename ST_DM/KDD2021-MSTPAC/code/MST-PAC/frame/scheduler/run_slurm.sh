#!/bin/bash 

root_dir=$(cd $(dirname "$0")/../../; pwd)

function main() {
    local config_file=$1
    
    local platform=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                     --conf_file ${config_file} --conf_name platform)
    local afs_host=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                     --conf_file ${config_file} --conf_name afs_host)
    local afs_user=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                     --conf_file ${config_file} --conf_name afs_user)
    local afs_passwd=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                     --conf_file ${config_file} --conf_name afs_passwd)
    local afs_mount_path=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name afs_mount_path)

    local job_name=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name job_name)
    local queue_name=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name queue_name)
    local nodes_num=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name nodes_num)
    local gpu_per_node_num=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name gpu_per_node_num)
    local workers_per_node_num=$(${fluid_bin} ${root_dir}/utils/ini_parser.py \
                      --conf_file ${config_file} --conf_name workers_per_node_num)

    echo $platform $afs_host $afs_user $afs_passwd $queue_name ${nodes_num} ${workers_per_node_num} >&2

    #HGCP_CLIENT_BIN=$HOME/.hgcp/software-install/HGCP_client/bin/submit
    HGCP_CLIENT_BIN=submit
    ${HGCP_CLIENT_BIN} \
            --hdfs $afs_host \
            --hdfs-user $afs_user \
            --hdfs-passwd $afs_passwd \
            --hdfs-path $afs_mount_path \
            --file-dir ./ \
            --job-name $job_name \
            --queue-name $queue_name \
            --num-nodes $nodes_num \
            --num-task-pernode $workers_per_node_num \
            --gpu-pnode $gpu_per_node_num \
            --time-limit 0 \
            --remarks $job_name"_remark" \
            --job-script ./frame/scheduler/slurm/job.sh
}

main "$@"
