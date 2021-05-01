#!/bin/bash
set -x

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95

export GLOG_v=1

if [ $# != 4 ];then
    echo "USAGE: sh script/run_dual_encoder_train.sh \$TRAIN_SET \$MODEL_PATH \$epoch \$nodes_count"
    exit 1
fi

TRAIN_SET=$1
MODEL_PATH=$2
epoch=$3
node=$4

CHECKPOINT_PATH=output
if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

lr=3e-5
batch_size=512
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
data_size=$[$save_steps*$batch_size*$node]
new_save_steps=$[$save_steps*$epoch/2]

python -m paddle.distributed.launch \
    --cluster_node_ips $(hostname -i) \
    --node_ip $(hostname -i) \
    --log_dir log \
    ./src/train_de.py \
                   --is_distributed false \
                   --use_recompute true \
                   --use_mix_precision true \
                   --use_cross_batch true \
                   --use_lamb false \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test false \
                   --batch_size ${batch_size} \
                   --train_data_size ${data_size} \
                   --train_set ${TRAIN_SET} \
                   --test_save ${CHECKPOINT_PATH}/score \
                   --use_fast_executor true \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --save_steps ${new_save_steps} \
                   --validation_steps ${new_save_steps} \
                   --learning_rate ${lr} \
                   --epoch ${epoch} \
                   --q_max_seq_len 32 \
                   --p_max_seq_len 128 \
                   --init_pretraining_params ${MODEL_PATH} \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/base/ernie_config.json \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.1 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   1>>log/train.log 2>&1

