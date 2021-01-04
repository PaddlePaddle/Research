#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python3  eval.py --use_gpu \
                                       --cfg ./configs/pspnet_res101_cityscapes.yaml 

#CUDA_VISIBLE_DEVICES=7 python3  eval.py --use_gpu \
#                                       --cfg ./configs/pspnet_res101_pascal_context.yaml 

#CUDA_VISIBLE_DEVICES=7 python3  eval.py --use_gpu \
#                                       --cfg ./configs/glore_res101_cityscapes.yaml 

#CUDA_VISIBLE_DEVICES=7 python3  eval.py --use_gpu \
#                                       --cfg ./configs/glore_res101_pascal_context.yaml 

