#!/bin/sh
export CUDA_VISIBLE_DEVICES=$2
export FLAGS_fraction_of_gpu_memory_to_use=0.1

~/apps/python3.7/bin/python tools/infer.py -c configs/aicity2020-track4/faster_rcnn_se154_vd_fpn_s1x_track4_new_anchors.yml --infer_dir $1 --output_dir $3

