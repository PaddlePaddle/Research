#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

~/apps/python3.7/bin/python tools/eval.py -c configs/aicity2020-track4/faster_rcnn_se154_vd_fpn_s1x_track4_new_anchors.yml -o weights=output/track4_faster_rcnn_se154_vd/50000
~/apps/python3.7/bin/python tools/eval.py -c configs/aicity2020-track4/faster_rcnn_se154_vd_fpn_s1x_track4.yml -o weights=output/track4_faster_rcnn_se154_vd/50000
