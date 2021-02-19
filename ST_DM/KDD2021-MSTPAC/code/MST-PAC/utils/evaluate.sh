#!/bin/bash 
function main() {
    infer_data="$1"
    ckpt_version="$2"
    #eval_cols="$3"
    echo "[TRACE] $(date) evaluate infer_data is ${infer_data}, checkpoint_version is ${ckpt_version}" >&2

    #startup script for eval
    cat ${infer_data} | awk 'BEGIN {FS="\t"; OFS="\t"} {print $3,$1,$5}' > ${infer_data}_for_eval
    echo "[TRACE] $(date) startup script for eval eval_data is ${infer_data}_for_eval, checkpoint_version is ${ckpt_version}" >&2
    sh utils/model_eval/eval_lite.sh ${infer_data}_for_eval $ckpt_version
    return 0
}

main "$@"
