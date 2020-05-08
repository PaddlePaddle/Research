export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memeory_to_use=0.1

python -u tools/infer.py -c configs_aicity/faster_rcnn_r50_fpn_1x.yml \
	                -o weights=./model/res50_80000 \
         			--infer_dir demo \
                    --output_dir check
