#!/bin/bash

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1

export DATA_PATH=./data/
export MODEL_PATH=./pretrained_lm/ernie2_base_en

lr=3e-5
batch_size=4
epoch=60
with_ent_structure=true

CUDA_VISIBLE_DEVICES=0 python ./run_ssan.py \
       --use_cuda true                                                   \
       --use_fast_executor ${e_executor:-"true"}                         \
       --tokenizer ${TOKENIZER:-"FullTokenizer"}                         \
       --use_fp16 ${USE_FP16:-"false"}                                   \
       --do_train true                                                   \
       --do_val true                                                     \
       --do_test false                                                   \
       --with_ent_structure ${with_ent_structure}                        \
       --model_path ${MODEL_PATH}                                        \
       --init_checkpoint ${MODEL_PATH}/params                            \
       --data_path ${DATA_PATH}                                          \
       --batch_size ${batch_size}                                        \
       --learning_rate ${lr}                                             \
       --warmup_proportion 0.1                                           \
       --epoch ${epoch}                                                  \
       --save_checkpoints ./checkpoints                                  \
       --max_seq_len 512                                                 \
       --max_ent_cnt 42                                                  \
       --num_labels 97                                                   \
       --random_seed 42                                                  \
       --skip_steps 50                                                   \
       --num_iteration_per_drop_scope 1                                  \
       --verbose true                                                    \
       --weight_decay  0.0