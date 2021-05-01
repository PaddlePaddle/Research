wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V1/data_train.tar.gz
tar -zxf data_train.tar.gz
rm -rf data_train.tar.gz

mkdir corpus
cd corpus

wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz

wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/nq.tar.gz
tar -zxf nq.tar.gz
rm -rf nq.tar.gz

wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/augment.tar.gz
tar -zxf augment.tar.gz
rm -rf augment.tar.gz
cd ..

for fold in marco nq;do
    cat data_train/${fold}_de1_denoise.tsv data_train/${fold}_unlabel_de2_denoise.tsv > data_train/${fold}_merge_de2_denoise.tsv 

    paste -d'\t' corpus/${fold}/para.title.txt corpus/${fold}/para.txt > corpus/${fold}/tp.tsv
    awk -F'\t' '{print "-\t"$2"\t"$4"\t0"}' corpus/${fold}/tp.tsv > corpus/${fold}/tp.tsv.format
    total_cnt=`cat corpus/${fold}/tp.tsv | wc -l`
    part_cnt=$[$total_cnt/8+1]
    mkdir corpus/${fold}/para_8part
    split -d -${part_cnt} corpus/${fold}/tp.tsv.format corpus/${fold}/para_8part/part-
    rm -rf corpus/${fold}/tp.tsv*
done

for file in corpus/marco/train.query.txt corpus/marco/dev.query.txt corpus/nq/train.query.txt corpus/nq/test.query.txt corpus/augment/orcas_yahoo_nq.query.txt corpus/augment/mrqa.query.txt;do
    awk -F'\t' '{print $2"\t-\t-\t0"}' $file > $file.format
done
