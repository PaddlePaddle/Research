#!/bin/bash
set -x

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
#export CUDA_VISIBLE_DEVICES=0

export GLOG_v=1

if [ $# != 2 ];then
    echo "USAGE: sh script/run_cross_encoder_test.sh \$TEST_SET \$MODEL_PATH"
    exit 1
fi

TASK_DATA=$1
MODEL_PATH=$2
batch_size=128

if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

python -u ./src/train_ce.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size ${batch_size} \
                   --init_checkpoint ${MODEL_PATH} \
                   --test_set ${TASK_DATA} \
                   --test_save output/${TASK_DATA}.score \
                   --max_seq_len 160 \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/large/ernie_config.json \
                   1>>log/test.log 2>&1

