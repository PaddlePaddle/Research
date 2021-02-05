#!/bin/bash

shell_path=$(cd $(dirname "$0"); pwd)
PYTHON=/home/map/tools/jumbo/bin/python4.8

out_file=score.tmp
eval_dest=score.$ckpt_version

statistic() {
    cat $1 | awk -F'\t'  '{if (length($1)>0 && $3 != "nan") {print $0}}' | sort -t$'\t' -k 1,1 > $out_file 

    #get auc
    cut -f2,3 $out_file | ${shell_path}/calc_auc

    #get pairloss
    cat $out_file | ${shell_path}/eval_pairloss

    #get nixu 
    cat $out_file | sh ${shell_path}/calc_nixu.sh

    #pos:neg order ratio for lego
    cat $out_file | awk -F'\t' '{printf("%s\t%s\t%s\n", $1, $3, $2)}' > score.lego.$ckpt_version
    sh /home/map/tools/lego/lego_mpi_release_float_sse_v5.1.9.0/scripts/multi_thread_eval.sh score.lego.$ckpt_version

    #ndcg and mrr
    $PYTHON ${shell_path}/top_satisfy.py $out_file 
    $PYTHON ${shell_path}/mrr_ndcg.py $out_file 
}

statistic $1 > $eval_dest
auc=`grep auc: $eval_dest | awk -F':' '{print $2}'`
pairloss=`grep eval_pairloss: $eval_dest | awk -F':' '{print $2}'`
nixu=`head -n 4 $eval_dest | tail -n 1`
pos_neg_ratio=`grep =:=: $eval_dest | awk '{print $2}'`
mrr1=`grep "mrr@1" $eval_dest | awk '{print $2}'`
mrr3=`grep "mrr@3" $eval_dest | awk '{print $2}'`
mrr5=`grep "mrr@5" $eval_dest | awk '{print $2}'`
ndcg1=`grep "ndcg@1" $eval_dest | awk '{print $2}'`
ndcg3=`grep "ndcg@3" $eval_dest | awk '{print $2}'`
ndcg5=`grep "ndcg@5" $eval_dest | awk '{print $2}'`
top_satisfy1=`grep "top 1 satisfy" $eval_dest | awk '{print $4}'`
top_satisfy3=`grep "top 3 satisfy" $eval_dest | awk '{print $4}'`
top_satisfy5=`grep "top 5 satisfy" $eval_dest | awk '{print $4}'`

echo -e "auc\tpairloss\tnixu\tpos_neg_ratio\tmrr1\tmrr3\tmrr5\tndcg1\tndcg3\tndcg5\ttop_satisfy1\ttop_satisfy3\ttop_satisfy5"
echo -e "$auc\t$pairloss\t$nixu\t$pos_neg_ratio\t$mrr1\t$mrr3\t$mrr5\t$ndcg1\t$ndcg3\t$ndcg5\t$top_satisfy1\t$top_satisfy3\t$top_satisfy5"
