set -eux

export BATCH_SIZE=8
export LR=2e-5
export EPOCH=12
export SAVE_STEPS=10000

CUDA_VISIBLE_DEVICES=0 python run_duie.py \
                   --seed 42 \
                   --model_name_or_path ernie-1.0 \
                   --do_train \
                   --data_path ./data \
                   --max_seq_length 128 \
                   --batch_size $BATCH_SIZE \
                   --num_train_epochs $EPOCH \
                   --learning_rate $LR \
                   --warmup_ratio 0.06 \
                   --logging_steps 50 \
                   --save_steps $SAVE_STEPS \
                   --output_dir ./checkpoints