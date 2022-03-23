CUDA_VISIBLE_DEVICES=2 # device index
DIR='clause'
SRC=$1
PRED=$2
python -u translate.py -model saved/$DIR/best/model_step_500000.pt -src saved/$DIR/$SRC -output saved/$DIR/$PRED -replace_unk -verbose --dynamic_dict