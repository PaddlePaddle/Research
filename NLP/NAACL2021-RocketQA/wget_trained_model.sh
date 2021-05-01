mkdir checkpoint
cd checkpoint

for filename in ernie_base_twin_init.tar.gz ernie_large_en.tar.gz marco_cross_encoder_large.tar.gz marco_dual_encoder_v0.tar.gz marco_dual_encoder_v2.tar.gz nq_cross_encoder_large.tar.gz nq_dual_encoder_v2.tar.gz;do
    wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V1/checkpoint/${filename}
    tar -zxf ${filename}
    rm -rf ${filename}
done

cd ..
