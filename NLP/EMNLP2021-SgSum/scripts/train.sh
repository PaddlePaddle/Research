#!/usr/bin/env bash
set -eux

source ./slurm/env_local.sh
source ./slurm/utils.sh
source ./model_config/roberta_graphsum_model_conf

rm -rf ./log
mkdir ./log

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export EVAL_SCRIPT_LOG=./log/eval.log
export GLOG_v=3

# check
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3 \
                --nproc_per_node 4"

python -u ./src/launch.py ${distributed_args} \
    ./src/run_roberta.py --model_name "multigraphsum" \
               --use_cuda true \
               --is_distributed true \
               --use_fast_executor ${e_executor:-"true"} \
               --use_fp16 ${use_fp16:-"false"} \
               --use_dynamic_loss_scaling ${use_fp16} \
               --init_loss_scaling ${loss_scaling:-128} \
               --weight_sharing true \
               --do_train true \
               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size ${BATCH_SIZE} \
               --in_tokens true \
               --stream_job ${STREAM_JOB:-""} \
               --init_pretraining_params ${MODEL_PATH} \
               --train_set ${TASK_DATA_PATH}/train \
               --dev_set ${TASK_DATA_PATH}/valid \
               --test_set ${TASK_DATA_PATH}/test \
               --config_path ./model_config/roberta_graphsum_config.json \
               --checkpoints ./model_checkpoints \
               --decode_path ./roberta_results \
               --lr_scheduler ${lr_scheduler} \
               --save_steps ${SAVE_STEPS} \
               --weight_decay ${WEIGHT_DECAY} \
               --warmup_proportion ${WARMUP_PROP} \
               --warmup_steps ${WARMUP_STEPS} \
               --validation_steps ${VALID_STEP} \
               --epoch ${EPOCH} \
               --max_para_num 100 \
               --max_para_len 768 \
               --max_tgt_len 300 \
               --max_doc_num 20 \
               --candidate_sentence_num 10 \
               --selected_sentence_num 9 \
               --block_trigram false \
               --graph_type "similarity" \
               --len_penalty 0.6 \
               --report_rouge true \
               --learning_rate ${LR_RATE} \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/roberta_graphsum.log" \
               --random_seed 1 > log/lanch.log 2>&1
