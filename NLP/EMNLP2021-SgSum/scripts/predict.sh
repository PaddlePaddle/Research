#!/usr/bin/env bash
set -eux

source ./slurm/env_local.sh
source ./slurm/utils.sh
source ./model_config/roberta_graphsum_model_conf

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export CUDA_VISIBLE_DEVICES="0"


python -u ./src/run_roberta.py \
               --model_name "multigraphsum" \
               --use_cuda true \
               --is_distributed false \
               --use_multi_gpu_test false \
               --use_fast_executor ${e_executor:-"true"} \
               --use_fp16 ${use_fp16:-"false"} \
               --use_dynamic_loss_scaling ${use_fp16} \
               --init_loss_scaling ${loss_scaling:-128} \
               --weight_sharing true \
               --do_train false \
               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size ${BATCH_SIZE} \
               --in_tokens true \
               --stream_job ${STREAM_JOB:-""} \
               --init_pretraining_params ${MODEL_PATH:-""} \
               --train_set ${TASK_DATA_PATH}/train \
               --dev_set ${TASK_DATA_PATH}/valid \
               --test_set ${TASK_DATA_PATH}/test \
               --config_path model_config/roberta_graphsum_config.json \
               --init_checkpoint model_checkpoints/ \
               --decode_path ./roberta_results/ \
               --lr_scheduler ${lr_scheduler} \
               --save_steps 10000 \
               --weight_decay ${WEIGHT_DECAY} \
               --warmup_steps ${WARMUP_STEPS} \
               --validation_steps 20000 \
               --epoch 100 \
               --max_para_num 100 \
               --max_para_len 768 \
               --max_tgt_len 300 \
               --max_out_len 300 \
               --min_out_len 200 \
               --candidate_sentence_num 10 \
               --selected_sentence_num 9 \
               --beam_size 5 \
               --graph_type "similarity" \
               --len_penalty 0.6 \
               --block_trigram true \
               --report_rouge true \
               --learning_rate ${LR_RATE} \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/cnndm_test.log" \
               --report_orcale false \
               --random_seed 1 > log/lanch.log 2>&1
