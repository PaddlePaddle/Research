wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/PAIR/models.tar.gz
tar -zxf models.tar.gz
cd models

for filename in ernie_base_twin_init.tar.gz marco_finetuned_encoder.tar.gz nq_finetuned_encoder.tar.gz;do
    tar -zxf ${filename}
    rm -rf ${filename}
done

cd ..
mv models checkpoint
