echo "start to extract frames from videos..."
starttime=`date +%s`
#get video list
videos_list=()
videos_list+=("$@")
echo "total video num : ${#videos_list[@]}"

tmp_fifofile="$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile

thread_num=31
data_root=$(pwd)"/AIC20_track1/Dataset_A"

for ((i=0;i<${thread_num};i++)); do
    echo
done >&6

#ffmpeg extract frames
for videoname in "${videos_list[@]}"; do
    read -u6
    {
        savepath="./imageset/${videoname}"
        mkdir -p ${savepath}
        ffmpeg -i ${data_root}/${videoname}.mp4 -q:v 2 ${savepath}/%05d.jpg -hide_banner
    	echo "$videoname : $(ls -lR $savepath|grep "^-"| wc -l) frames"
        echo >&6
    } &
done
wait

endtime=`date +%s`
echo "Extracting frames costs : `expr $endtime - $starttime` s"
echo "Done."
exec 6>&- 
exit

