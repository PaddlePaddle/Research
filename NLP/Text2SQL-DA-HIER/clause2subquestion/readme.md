## clone open-nmt

    git clone https://github.com/OpenNMT/OpenNMT-py.git
    cd OpenNMT-py
    pip install -e .


## Procedure

### preprocess

    onmt_build_vocab -config spider/spider.yaml

### train

    CUDA_VISIBLE_DEVICES=2 onmt_train -config spider/spider.yaml --copy_attn_force --copy_attn

### predict

    onmt_translate -model toy-ende/run/model_step_1000.pt -src toy-ende/src-test.txt -output toy-ende/pred_1000.txt -gpu 0 -verbose

