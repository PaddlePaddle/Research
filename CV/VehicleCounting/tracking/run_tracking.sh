echo "start to do online tracking..."
starttime=`date +%s`
videos_list=()
videos_list+=("$@")

root=$(dirname "$PWD")
#echo $root
tmp_fifofile="$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile

thread_num=31
for ((i=0;i<${thread_num};i++)); do
    echo
done >&6

for videoname in "${videos_list[@]}"; do
    read -u6
    {
        ./build/aicity_task1_tracking $videoname $root 0
	echo >&6
    } &
done
wait

endtime=`date +%s`
echo "online tracking costs: `expr $endtime - $starttime` s"
echo "Done."
exec 6>&-
exit
