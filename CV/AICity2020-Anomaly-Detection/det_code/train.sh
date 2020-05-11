export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 tools/train.py -c configs/aicity2020-track4/faster_rcnn_se154_vd_fpn_s1x_track4_new_anchors.yml
