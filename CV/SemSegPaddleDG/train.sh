#!/bin/bash

# single_card:
CUDA_VISIBLE_DEVICES=7 python3 train.py  --use_gpu \
                                  --cfg ./configs/pspnet_res101_cityscapes.yaml 
# multi_card:
#python3  -m paddle.distributed.launch --selected_gpus=6,7 train.py \
#                                  --use_gpu --cfg configs/pspnet_res101_cityscapes.yaml

