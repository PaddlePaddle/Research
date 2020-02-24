#!/bin/bash

:<<!
训练模型并评估
!

HERE=$(readlink -f "$(dirname "$0")")
cd ${HERE}/..

DATA_DIR=${HERE}/../../data
PRETRAIN_MODEL=${HERE}/../../model/ERNIE_1.0_max-len-512
SAVE_MODEL=${HERE}/../../save_model
DICT=${HERE}/../../dict
GPUID=0

RESULT_DIR=${HERE}/../../result


# 目录不存在先创建
if [[ ! -d ${SAVE_MODEL} ]]; then
    mkdir ${SAVE_MODEL}
fi

TRIGGER_SAVE_MODEL=${SAVE_MODEL}/trigger
ROLE_SAVE_MODEL=${SAVE_MODEL}/role

# STEP 1: 训练触发词模型
echo -e "**********TRIGGER TRAIN START**********"
if [[ -d ${TRIGGER_SAVE_MODEL}/final_model ]]; then
    echo "trigger model is trained"
else
    sh script/train_event_trigger.sh ${GPUID} ${DATA_DIR} ${TRIGGER_SAVE_MODEL} ${PRETRAIN_MODEL} ${DICT}
fi
echo "**********TRIGGER TRAIN END**********"


# STEP 2: 预测句子触发词识别的结果
echo -e "\n\n**********TRIGGER PREDICT START**********"
if [[ -e ${TRIGGER_SAVE_MODEL}/pred_trigger.json ]]; then
    echo "trigger predict is finished"
else
    sh script/predict_event_trigger.sh ${GPUID} ${DATA_DIR} ${PRETRAIN_MODEL} ${TRIGGER_SAVE_MODEL}/final_model ${DICT}
fi
echo "**********TRIGGER PREDICT END**********"


# STEP 3: 训练论元角色模型
echo -e "\n\n**********ROLE TRAIN START**********"
if [[ -d ${ROLE_SAVE_MODEL}/final_model ]]; then
    echo "role model is trained"
else
    sh script/train_event_role.sh ${GPUID} ${DATA_DIR} ${ROLE_SAVE_MODEL} ${PRETRAIN_MODEL} ${DICT}
fi
echo "**********ROLE TRAIN END**********"


# STEP 4: 预测句子中存在的论元角色模型
echo -e "\n\n**********ROLE PREDICT START**********"
if [[ -e ${ROLE_SAVE_MODEL}/pred_role.json ]]; then
    echo "role predict is finished"
else
    sh script/predict_event_role.sh ${GPUID} ${DATA_DIR} ${PRETRAIN_MODEL} ${ROLE_SAVE_MODEL}/final_model ${DICT}
fi
echo "**********ROLE PREDICT END**********"


# STEP 5: 预测结果处理成评估需要的格式
echo -e "\n\n**********RESULT START**********"
python predict_eval_process.py test_data_2_eval ${DATA_DIR}/test.json ${RESULT_DIR}/gold.json # 测试集转化为评估格式
TRIGGER_PRED_FILE=${TRIGGER_SAVE_MODEL}/pred_trigger.json
ROLE_PRED_FILE=${ROLE_SAVE_MODEL}/pred_role.json
if [[ ! -e ${TRIGGER_PRED_FILE} || ! -e ${ROLE_PRED_FILE} ]]; then
    echo "check trigger or role extract is finished"
else
    python predict_eval_process.py predict_data_2_eval ${TRIGGER_PRED_FILE} ${ROLE_PRED_FILE} ${DICT}/event_schema.json ${RESULT_DIR}/pred.json
fi
echo "**********RESULT END**********"
