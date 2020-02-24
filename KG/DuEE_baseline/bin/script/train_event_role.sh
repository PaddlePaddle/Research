#!/bin/bash

set -eux

HERE=$(readlink -f "$(dirname "$0")")
cd ${HERE}/..

export CUDA_VISIBLE_DEVICES=$1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.3

DATA_DIR=$2
SAVE_CKPT=$3
MODEL_PATH=$4
DICT=$5

python run_event_role.py --use_cuda true\
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --trigger_pred_save_path ${SAVE_CKPT}/pred_role.json \
                   --chunk_scheme "IOB" \
                   --label_map_config ${DICT}/vocab_roles_label_map.txt \
                   --train_set ${DATA_DIR}/train.json \
                   --dev_set ${DATA_DIR}/dev.json \
                   --test_set ${DATA_DIR}/test.json \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --checkpoints ${SAVE_CKPT} \
                   --save_steps 500 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.1 \
                   --validation_steps 100 \
                   --use_fp16 false \
                   --epoch 6 \
                   --max_seq_len 300 \
                   --crf_learning_rate 0.2 \
                   --learning_rate 2e-4 \
                   --skip_steps 20 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1
