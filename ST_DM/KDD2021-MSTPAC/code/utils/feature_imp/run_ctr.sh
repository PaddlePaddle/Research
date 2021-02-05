#!/bin/bash
#run with nohup sh run_ctr.sh&, in current dir

PYTHON=/home/map/rd/wangjiuyang/program/paddle/bin/gpu/1.6.1/cuda9/paddle_release_home/python/bin/python4.8
TEST=feature_important_eval
WORK_PATH=`pwd`
TMP_CONF=${WORK_PATH}/tmp/conf
TMP_SCORE=${WORK_PATH}/tmp/score
TMP_TEST_DATA=${WORK_PATH}/tmp/replace_file

#############vars can be modify begin
#one part of test data
ORG_TEST=part-00000
#modify paddle-frame predict conf, which test dataset must use the $TEST
#must use CPU to predict, very important
CONF=/home/map/rd/chenyu14/project/paddle-frame/baidu/mapsearch/paddle-frame/conf/deep_fm/deep_fm_pair.feat_imp.conf
#conf to confirm column need to evaluate
REPLACE_CONF=common.conf
#and a head file must be prepared in this dir, format as "sample_column_index\thead_name"
###############vars can be modify end

rm -rf ${WORK_PATH}/tmp
mkdir ${WORK_PATH}/tmp
mkdir $TMP_CONF
mkdir $TMP_SCORE
mkdir $TMP_TEST_DATA

#get base score
cp $ORG_TEST $TEST
cd ../../
sh run.sh -c $CONF -m predict 1>${TMP_SCORE}/base.qid_label_sore 2>/dev/null
cd -
tmp_score=`sh ../model_eval/eval_one.sh ${TMP_SCORE}/base.qid_label_sore`
echo "base	"$tmp_score > all_score
date

#format all test_data
$PYTHON replace_column.py $ORG_TEST $REPLACE_CONF $TMP_TEST_DATA

#predict parallel use CPU
files=`ls $TMP_TEST_DATA`
cd ../../
for file in $files;
do
    new_path="tmp\/replace_file\/"${file}
    sed "s/${TEST}/${new_path}/" $CONF > ${TMP_CONF}/${file}.conf
    sh run.sh -c ${TMP_CONF}/${file}.conf -m predict 1>${TMP_SCORE}/${file}.qid_label_sore 2>/dev/null&
done
wait
date

#eval qid_label_score
cd ${WORK_PATH}
for file in $files;
do
    tmp_score=`sh ../model_eval/eval_one.sh ${TMP_SCORE}/${file}.qid_label_sore`
    echo $file"	"$tmp_score >> all_score
done

#output final feature_imp
$PYTHON union_feature_name.py all_score head > final_feature_imp
