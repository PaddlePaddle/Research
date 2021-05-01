#!/bin/bash
set -eux

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CUDA_VISIBLE_DEVICES=$1

if [ $# != 4 ];then
    echo "USAGE: sh script/run_dual_encoder_inference.sh \$card_id \$part \$MODEL_PATH \$DATA_PATH"
    exit 1
fi

batch_size=256
part=$2
MODEL_PATH=$3
DATA_PATH=$4

if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi


if [ $part == 'q' ];then
    TASK_DATA_PATH=$DATA_PATH
    out_file_name=query.emb
    out_item=0
else
    TASK_DATA_PATH=${DATA_PATH}/para_8part/part-0${part}
    out_file_name=para.index.part${part}
    out_item=1
fi
test_data_cnt=`cat $TASK_DATA_PATH | wc -l`

python -u ./src/inference_de.py                  \
       --use_cuda true                                                                  \
       --use_fast_executor ${e_executor:-"true"}                                        \
       --do_train false                                                                  \
       --do_val false                                                                   \
       --do_test true                                                                  \
       --batch_size $batch_size                                                         \
       --init_checkpoint ${MODEL_PATH} \
       --test_set ${TASK_DATA_PATH} \
       --test_save output/test_out.tsv \
       --output_item ${out_item} \
       --output_file_name ${out_file_name} \
       --test_data_cnt $test_data_cnt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 128                                                              \
       --vocab_path config/vocab.txt \
       --ernie_config_path config/base/ernie_config.json \
       1>>log/test.log.${part} 2>&1
