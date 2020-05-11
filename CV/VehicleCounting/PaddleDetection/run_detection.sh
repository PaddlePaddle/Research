#!/bin/bash
starttime=$(date "+%s")
echo "start running detection..."
out_root='output'
videos_list=()
videos_list+=("$@")

tmp_fifofile="$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile

thread_num=16
gpu_num=4

for ((i=1;i<=${thread_num};i++)); do
    echo $i
done >&6

for videoname in "${videos_list[@]}"; do
    read -u6 t_id
    {
	gpu_id=`expr $t_id % $gpu_num`
	echo "run $videoname: thread $t_id gpu: $gpu_id"
        sh frcnn_res50_infer.sh ../imageset/$videoname $gpu_id $out_root
        echo $t_id >&6	
    } &
done
wait

endtime=`date +%s`
echo "Detect all frames costs: `expr $endtime - $starttime` s"
echo "Done."
exec 6>&-
exit
