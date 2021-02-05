#!/bin/sh
CUR_PATH=$(echo ${BASH_SOURCE} | xargs dirname)
ROOT_PATH=$(readlink -f $CUR_PATH/../../)

cluster_list="mced-0-yq01-k8s-gpu-p40-8:4 mced-1-yq01-k8s-gpu-p40-8:4 mced-2-yq01-k8s-gpu-p40-8:8 mced-16g-0-yq01-k8s-gpu-v100-8:4 mced-16g-1-yq01-k8s-gpu-v100-8:4 mced-16g-2-yq01-k8s-gpu-v100-8:8"
for cluster in ${cluster_list}; do 
    sh ${CUR_PATH}/deploy_k8s.sh ${cluster%:*} ${cluster#*:}
done

exit 0
