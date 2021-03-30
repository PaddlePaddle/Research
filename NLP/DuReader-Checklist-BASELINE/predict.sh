#!/bin/bash
export PYTHONIOENCODING=utf-8

if [ -z "$CUDA_VISIBLE_DEVICES" ];then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -u src/run.py \
    --model_type ernie \
    --max_seq_length 512 \
    --batch_size 4 \
    --logging_steps 50 \
    --max_answer_length 512 \
    --output_dir output \
    --version_2_with_negative \
    --do_pred \
    --cls_threshold 0.7 \
    $@
