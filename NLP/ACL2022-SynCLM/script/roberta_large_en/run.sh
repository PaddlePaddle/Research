#!/usr/bin/env bash
source env_local/env.sh
export FLAGS_fuse_parameter_memory_size=64
export PADDLE_PORT=8454

distributed_args="--node_ips 127.0.0.1 \
                --node_id 0 \
                --current_node_ip 127.0.0.1 \
                --selected_gpus 0,1 \
                --split_log_path ./output/log \
                --nproc_per_node 1"
mkdir -p ./output/log
python3 -u ./src/launch.py ${distributed_args} \
    ./src/pretrain.py --use_cuda "True" \
                --is_distributed "True" \
                --visualdl_log False \
                --weight_sharing "True" \
                --use_fast_executor "True" \
                --use_fuse "True" \
                --nccl_comm_num 2 \
                --use_hierarchical_allreduce "True" \
                --in_tokens "True" \
                --batch_size 6250 \
                --train_filelist "./data/pretrain/train_filelist" \
                --valid_filelist "./data/pretrain/valid_filelist" \
                --random_seed 666 \
                --lr_scheduler "linear_warmup_decay" \
                --num_train_steps 3500000 \
                --checkpoints "./output" \
                --use_fp16 "False" \
                --use_dynamic_loss_scaling "False" \
                --init_loss_scaling "12800" \
                --beta1 0.9 \
                --beta2 0.98 \
                --epsilon 1e-06 \
                --save_steps 50000 \
                --validation_steps 10000 \
                --init_checkpoint "./model_files/roberta_large_en" \
                --init_step 0 \
                --roberta_vocab_file "./model_files/dict/roberta_large_en.vocab.txt" \
                --encoder_json_file "./model_files/dict/roberta_large_en.encoder.json" \
                --vocab_bpe_file "./model_files/dict/roberta_large_en.vocab.bpe" \
                --synclm_config_path "./model_files/config/roberta_large_en.json" \
                --learning_rate 3e-5 \
                --warmup_steps 30000 \
                --weight_decay 0.01 \
                --max_seq_len 512 \
                --num_iteration_per_drop_scope 100 \
                --att_layer -1 \
                --tree_max_sub_num 10 \
                --tree_max_neg_num 30 \
                --phrase_max_neg_num 10 \
                --skip_steps 100 >> ./output/lanch.log 2>&1

if [[ $? -ne 0 ]]; then
    echo "run failed"
    exit 1
fi

exit 0