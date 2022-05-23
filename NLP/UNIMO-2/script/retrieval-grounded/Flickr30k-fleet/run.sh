#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
# config env
source ${MYDIR}/model_conf

source ./env_local/env.sh
source ./env_local/utils.sh

set -x
export PADDLE_WITH_GLOO=0
export GLOG_v=0
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
#export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

timestamp=`date "+%Y%m%d-%H%M%S"`
echo $timestamp

# check
check_iplist

set -eu
output_dir=./output/${task}
log_dir=./output/${task}/log
rm -rf $output_dir
save_model_base_dir=$output_dir/save_model
mkdir -p $output_dir $log_dir $save_model_base_dir

e_executor=$(echo ${use_experimental_executor:-'True'} | tr '[A-Z]' '[a-z]')
use_fuse=$(echo ${use_fuse:-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    #MB
    export FLAGS_fuse_parameter_memory_size=64
fi

log_prefix=$seed"_"$epoch"_"$lr"_"$batch_size"."
eval_dir="${output_dir}/tmp/params.${seed}.${epoch}.${lr}.${batch_size}"
mkdir -p $eval_dir

if [[ ${save_checkpoints} == "True" ]]; then
    save_model_dir="${save_model_base_dir}/params.${seed}.${epoch}.${lr}.${batch_size}"
    mkdir -p $save_model_dir
fi

distributed_args="--ips $(hostname -i) \
                --gpus 0,1,2,3,4,5,6,7 \
                --log_dir $log_dir"

python3 -m paddle.distributed.launch ${distributed_args} \
    ./src/run_retrieval_grounded_fleet.py \
    --use_cuda "True" \
    --is_distributed ${is_distributed:-"True"} \
    --weight_sharing ${weight_sharing:-"True"} \
    --use_fuse ${use_fuse:-"True"} \
    --use_fast_executor ${e_executor:-"true"} \
    --use_fp16 ${use_fp16:-"false"} \
    --nccl_comm_num ${nccl_comm_num:-1} \
    --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
    --use_dynamic_loss_scaling ${use_fp16:-"False"} \
    --use_sigmoid ${use_sigmoid:-"False"} \
    --init_loss_scaling ${loss_scaling:-12800} \
    --beta1 ${beta1:-0.9} \
    --beta2 ${beta2:-0.98} \
    --epsilon ${epsilon:-1e-06} \
    --scale_circle ${scale_circle:-1.0} \
    --margin ${margin:-0.2} \
    --verbose true \
    --samples_num ${samples_num:-20} \
    --run_random ${run_random:-"False"} \
    --do_train ${do_train:-"True"} \
    --do_val ${do_val:-"True"} \
    --do_test ${do_test:-"True"} \
    --batch_size ${batch_size:-16} \
    --test_batch_size ${test_batch_size:-96} \
    --init_pretraining_params ${init_model:-""} \
    --train_image_caption ./data/Flickr30k/flickr30k-textids/train.ids \
    --dev_image_caption ./data/Flickr30k/flickr30k-textids/val.all.ids \
    --test_image_caption ./data/Flickr30k/flickr30k-textids/test.all.ids \
    --checkpoints ${save_model_dir:-""} \
    --save_checkpoints ${save_checkpoints:-"True"} \
    --save_steps ${save_steps:-1000} \
    --weight_decay ${weight_decay:-"0.1"} \
    --warmup_step ${warmup_step:-"1"} \
    --validation_steps ${validation_steps:-"100"} \
    --epoch $epoch \
    --max_seq_len ${max_len:-512} \
    --learning_rate ${lr:-"5e-6"} \
    --learning_rate_scale ${learning_rate_scale:-0.1} \
    --learning_rate_decay_epoch1 ${learning_rate_decay_epoch1:-24} \
    --learning_rate_decay_epoch2 ${learning_rate_decay_epoch2:-32} \
    --lr_scheduler ${lr_scheduler:-"scale_by_epoch_decay"} \
    --skip_steps ${skip_steps:-"50"} \
    --num_iteration_per_drop_scope 10 \
    --unimo_vocab_file ${vocab_file} \
    --encoder_json_file ${bpe_json} \
    --vocab_bpe_file ${bpe_file} \
    --unimo_config_path ${CONFIG_PATH} \
    --eval_mertrics ${eval_mertrics:-"recall@k"} \
    --eval_dir $eval_dir \
    --random_seed ${seed:-1} \
    --resolution ${resolution:-16} \
    --image_size ${image_size:-224} \
    --num_codebook ${num_codebook:-2048} \
    --grounding_method ${grounding_method:-'normal'} \
    --topk_value ${topk_value:-100} \
    --model_type ${model_type:-"grounded"} \
    --with_cmcl ${with_cmcl:-"True"} \
    --with_grounding_projection ${with_grounding_projection:-"True"} \
    --cmcl_share_parameters ${cmcl_share_parameters:-"True"} \
    --with_cmcl_projection ${with_cmcl_projection:-"True"} \
    --cmcl_score_weight ${cmcl_score_weight:-0.1} \
    --with_grounding_pos ${with_grounding_pos:-"False"} \
    --text_enc_layers ${text_enc_layers:-'0,1,2,3,4,5'} \
    --grounding_enc_layers ${grounding_enc_layers:-'6,7,8,9,10,11'} \
    --use_recompute ${use_recompute:-'False'} \
    >> $log_dir/${log_prefix}lanch.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi

python3 ./src/utils/new_stat_res.py --log_dir=$log_dir --key_words='workerlog.0'
exit 0
