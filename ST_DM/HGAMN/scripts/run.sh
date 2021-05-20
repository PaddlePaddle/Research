jobname=`basename ${0} | cut -d '.' -f 1`

python train.py --sage_mode "gcn" --model_type "cnn" \
           --ernie_config ./encoder_config/model.json \
           --vocab_path ./encoder_config/vocab.txt --batch_size 128 --use_cuda 1 \
           --num_workers 1 --output_path output_model/${jobname}  \
           --token_mode small \
           --epoch 5  \
           --graph_data ./graphs/poi-query.graph ./graphs/poi-poi.graph \
           --data_path ./raw_data/train.txt \
           --eval_path ./raw_data/test.txt \
           --learning_rate 1e-4 \
           --norm_score False \
           --scale_softmax False \
           --with_city True \
           --with_geo_id   True \

