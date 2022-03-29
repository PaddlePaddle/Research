
DATA='spider'
onmt_preprocess -train_src $DATA/train_src.txt -train_tgt $DATA/train_tgt.txt -valid_src $DATA/dev_src.txt -valid_tgt $DATA/dev_tgt.txt -save_data $DATA/demo -dynamic_dict
