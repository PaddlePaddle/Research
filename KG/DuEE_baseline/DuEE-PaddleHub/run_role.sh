#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.3

data_dir=$2
ckpt_dir=$3

python sequence_label.py --num_epoch 3 \
    --learning_rate 3e-5 \
    --data_dir ${data_dir} \
    --schema_path ${data_dir}/event_schema.json \
    --train_data ${data_dir}/train.json \
    --dev_data ${data_dir}/dev.json \
    --test_data ${data_dir}/dev.json \
    --predict_data ${data_dir}/test1.json \
    --do_train True \
    --do_predict True \
    --do_model role \
    --max_seq_len 256 \
    --batch_size 8 \
    --model_save_step 3000 \
    --eval_step 200 \
    --checkpoint_dir ${ckpt_dir}

