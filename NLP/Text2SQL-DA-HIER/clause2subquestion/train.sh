
SAVED=saved/clause_new
mkdir $SAVED
CUDA_VISIBLE_DEVICES=0 # device index
python -u train.py -data clause_all/demo \
        -save_model $SAVED/model \
        -world_size 1 \
        --input_feed 1 \
        -copy_attn \
        -copy_attn_force \
        --optim  'adam'\
        --learning_rate 0.001 \
        --train_steps 500000 \
        --early_stopping 20 \
        --valid_steps 10000 \
        --fix_word_vecs_enc \
        --gpu_ranks 0 >$SAVED/train.log 2>&1 &