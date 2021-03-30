#!/bin/bash
export PYTHONIOENCODING=utf-8

unset CUDA_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m paddle.distributed.launch --gpus "0" src/run.py \
    --model_type ernie \
    --model_name_or_path ernie-1.0 \
    --max_seq_length 512 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 50 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --max_answer_length 512 \
    --weight_decay 0.01 \
    --output_dir output \
    --version_2_with_negative \
    --do_train \
    --do_pred \
    --train_file dataset/train.json \
    --predict_file dataset/dev.json \
    --cls_threshold 0.7 \
    --device gpu \
    $@
