#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
# config env
source ${MYDIR}/model_conf

source ./env_local/env.sh
source ./env_local/utils.sh

set -eu
output_dir=./output/${task}
log_dir=./output/${task}/log
rm -rf $output_dir
save_model_dir=$output_dir/save_model
mkdir -p $output_dir $log_dir $save_model_dir

e_executor=$(echo ${use_experimental_executor:-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse:-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    #MB
    export FLAGS_fuse_parameter_memory_size=64
fi

# check
check_iplist

if [[ ${do_pred} == "True" ]]; then
    pred_save_prefix="${output_dir}/predict"
    mkdir -p $pred_save_prefix
fi

for seed in "${DD_RAND_SEED[@]}"; do
  echo "seed "$seed
  for epoch in "${EPOCH[@]}"; do
    echo "epoch "$epoch
    for lr in "${LR_RATE[@]}"; do
      echo "learning rate "$lr
      for bs in "${BATCH_SIZE[@]}"; do
        echo "batch_size "$bs
        log_prefix=$seed"_"$epoch"_"$lr"_"$bs"."
        if [[ ${do_pred} == "True" ]]; then
            pred_save="${pred_save_prefix}/test.${seed}.${epoch}.${lr}.${bs}"
        fi

        if [[ ${save_checkpoints} == "True" ]]; then
            save_model_dir="${save_model_dir}/params.${seed}.${epoch}.${lr}.${bs}"
            mkdir -p $save_model_dir
        fi

        if [[ ${bs} == "32" ]]; then
            validation_steps=10000
        fi

        distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3,4,5,6,7 \
                --split_log_path $log_dir \
                --log_prefix $log_prefix \
                --nproc_per_node 8"

        python3 -u ./src/launch.py ${distributed_args} \
            ./src/run_classifier_grounded.py --use_cuda "True" \
                   --is_distributed ${is_distributed:-"True"} \
                   --weight_sharing ${weight_sharing:-"True"} \
                   --use_fast_executor ${e_executor:-"true"} \
                   --use_fp16 ${use_fp16:-"false"} \
                   --nccl_comm_num ${nccl_comm_num:-1} \
                   --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                   --in_tokens ${in_tokens:-"false"} \
                   --use_dynamic_loss_scaling ${use_fp16} \
                   --use_modal_tag ${use_modal_tag:-"false"} \
                   --init_loss_scaling ${loss_scaling:-128} \
                   --beta1 ${beta1:-0.9} \
                   --beta2 ${beta2:-0.98} \
                   --epsilon ${epsilon:-1e-06} \
                   --verbose true \
                   --do_train ${do_train:-"True"} \
                   --do_val ${do_val:-"True"} \
                   --do_val_hard ${do_val_hard:-"False"} \
                   --do_test ${do_test:-"True"} \
                   --do_test_hard ${do_test_hard:-"False"} \
                   --do_pred ${do_pred:-"True"} \
                   --do_pred_hard ${do_pred_hard:-"False"} \
                   --do_diagnostic ${do_diagnostic:-"True"} \
                   --pred_save ${pred_save:-"./output/predict/test"} \
                   --batch_size ${bs:-16} \
                   --init_pretraining_params ${init_model:-""} \
                   --train_set ./data/$task/train.tsv \
                   --dev_set ./data/$task/m/dev.tsv \
                   --dev_hard_set ./data/$task/mm/dev.tsv \
                   --test_set ./data/$task/m/test.tsv \
                   --test_hard_set ./data/$task/mm/test.tsv \
                   --diagnostic_set ./data/$task/diagnostic.tsv \
                   --checkpoints $save_model_dir \
                   --save_checkpoints ${save_checkpoints:-"True"} \
                   --save_steps ${save_steps:-1000} \
                   --weight_decay ${weight_decay:-"0.1"} \
                   --warmup_proportion ${warmup_ratio:-"0.06"} \
                   --validation_steps ${validation_steps:-"100"} \
                   --epoch $epoch \
                   --max_seq_len ${max_len:-512} \
                   --learning_rate ${lr:-"5e-5"} \
                   --lr_scheduler ${lr_scheduler} \
                   --skip_steps ${skip_steps:-"10"} \
                   --num_iteration_per_drop_scope 10 \
                   --num_labels ${num_labels:-2} \
                   --roberta_vocab_file ${vocab_file} \
                   --encoder_json_file ${bpe_json} \
                   --vocab_bpe_file ${bpe_file} \
                   --vl_config_path ${CONFIG_PATH} \
                   --eval_mertrics ${eval_mertrics:-"simple_accuracy"} \
                   --model_type ${model_type:-"fleet"} \
                   --num_codebook ${num_codebook:-2048} \
                   --grounding_method ${grounding_method:-"topk"} \
                   --topk_value ${topk_value:-100} \
                   --with_grounding_projection ${with_grounding_projection:-"False"} \
                   --with_grounding_pos ${with_grounding_pos:-"False"} \
                   --text_enc_layers ${text_enc_layers:-'0,1,2,3,4,5'} \
                   --grounding_enc_layers ${grounding_enc_layers:-'6,7,8,9,10,11'} \
                   --random_seed ${seed:-1} >> $log_dir/${log_prefix}lanch.log 2>&1
      done
    done
  done
done

python ./src/utils/stat_res.py --log_dir=$log_dir --key_words='job.log' --line_prefix="Best validation result:" --final_res_file="final_res.m.txt"
python ./src/utils/stat_res.py --log_dir=$log_dir --key_words='job.log' --line_prefix="Best validation_hard result:" --final_res_file="final_res.mm.txt"

exit 0
