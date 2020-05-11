starttime=`date +%s`
videos_list=()
videos_list+=("$@")

tmp_fifofile="$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile

thread_num=31

for ((i=0;i<${thread_num};i++)); do
    echo
done >&6

for i in "${videos_list[@]}"; do
    read -u6
    {
	python counting.py ${i}
        echo >&6
    } &
done
wait

endtime=`date +%s`
echo "TIME : `expr $endtime - $starttime` s"
exec 6>&-
exit
