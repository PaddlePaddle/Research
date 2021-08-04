wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/PAIR/data.tar.gz
tar -zxf data.tar.gz
rm -rf data.tar.gz
mv data data_train

mkdir corpus
cd corpus

wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz

wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/nq.tar.gz
tar -zxf nq.tar.gz
rm -rf nq.tar.gz

cd ../

for fold in marco nq;do
    paste -d'\t' corpus/${fold}/para.title.txt corpus/${fold}/para.txt > corpus/${fold}/tp.tsv
    awk -F'\t' '{print "-\t"$2"\t"$4"\t0"}' corpus/${fold}/tp.tsv > corpus/${fold}/tp.tsv.format
    total_cnt=`cat corpus/${fold}/tp.tsv | wc -l`
    part_cnt=$[$total_cnt/8+1]
    mkdir corpus/${fold}/para_8part
    split -d -${part_cnt} corpus/${fold}/tp.tsv.format corpus/${fold}/para_8part/part-
    rm -rf corpus/${fold}/tp.tsv*
done

for file in corpus/marco/train.query.txt corpus/marco/dev.query.txt corpus/nq/train.query.txt corpus/nq/test.query.txt;do
    awk -F'\t' '{print $2"\t-\t-\t0"}' $file > $file.format
done
