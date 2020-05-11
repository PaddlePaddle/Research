starttime=`date +%s`
video_list_txt='../AIC20_track1/Dataset_A/list_video_id.txt'
video_id_list=()
while read -r line || [[ -n "$line" ]]; do
    stringarray=($line)
    video=${stringarray[1]}
    video_name=${video%%.*}
    video_id_list+=(${video_name})
done < ${video_list_txt}

tmp_fifofile="/tmp/$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile

thread_num=31

for ((i=0;i<${thread_num};i++)); do
    echo
done >&6

for i in "${video_id_list[@]}"; do
    read -u6
    {
        echo ${i}
	#python count.py ${i}
	python counting.py ${i}
        echo >&6
    } &
done
wait

endtime=`date +%s`
echo "TIME : `expr $endtime - $starttime` s"
exec 6>&-
exit
