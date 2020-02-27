#!/bin/bash

:<<!
准备用于训练的数据
!
here=$(readlink -f "$(dirname "$0")")
ROOT=${here}/..
cd ${ROOT}

# 把数据处理成训练集、测试集等
data_dir=${ROOT}/../data
echo -e "\n====data process start===="
if [[ ! -e ${data_dir}/train.json && ! -e ${data_dir}/test.json ]] ;then
    python data_process.py origin_events_process ${data_dir}/eet_events.json ${data_dir}
    echo "data process is finished"
else
    echo "data file exists"
fi
echo -e "====data process end===="

# 处理schema文件成标签
schema_dir=${ROOT}/../dict
echo -e "\n\n====event type process start===="
event_type_label_file=${schema_dir}/vocab_trigger_label_map.txt
if [[ ! -e ${event_type_label_file} ]]; then
    python data_process.py schema_event_type_process ${schema_dir}/event_schema.json ${event_type_label_file}
    echo "event type label process is finished"
else
    echo "event type label ${event_type_label_file} exists"
fi
echo -e "====event type process end===="

event_role_label_file=${schema_dir}/vocab_roles_label_map.txt
echo -e "\n\n====role process start===="
if [[ ! -e ${event_role_label_file} ]]; then
    python data_process.py schema_role_process ${schema_dir}/event_schema.json ${event_role_label_file}
    echo "role label label process is finished"
else
    echo "role label ${event_role_label_file} exists"
fi
echo -e "====role process end===="

# 下载预训练模型
cd ${ROOT}/../model
pretrain_model=ERNIE_1.0_max-len-512
echo -e "\n\n====pretrain model download start===="
if [[ ! -d ${pretrain_model} ]] ;then
    pretrain_model_tar=${pretrain_model}.tar.gz
    if [[ -e ${pretrain_model_tar} ]]; then
        rm ${pretrain_model_tar}
    fi
    wget https://ernie.bj.bcebos.com/${pretrain_model_tar} --no-check-certificate
    echo "download pretrain model finish"
    mkdir ${pretrain_model}
    tar -zxf ${pretrain_model_tar} -C ${pretrain_model}
    echo "unzip pretrain model finish"
else
    echo "pretrain model exists"
fi
echo -e "====pretrain model download end===="
cd ${ROOT}

