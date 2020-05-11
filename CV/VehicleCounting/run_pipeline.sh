#!/usr/bin/env bash
echo "Start to process AICity2020 task1:"
starttime=`date +%s`
echo "---------------------------------------------------"
#get all videos names
ROOT_PATH=$(pwd)
video_list_txt=$ROOT_PATH"/AIC20_track1/Dataset_A/list_video_id.txt"
#echo $video_list_txt
#get video list
videos_list=()
while read -r line || [[ -n "$line" ]]; do
    stringarray=($line)
    videoname=${stringarray[1]%%.*}
    videos_list+=(${videoname})
done < ${video_list_txt}

#extract all frames
sh extract_frames.sh "${videos_list[@]}"
echo "---------------------------------------------------"
#detection
cd PaddleDetection
rm -rf output
sh run_detection.sh "${videos_list[@]}"
cp -r output/det_results ../
echo "---------------------------------------------------"
cd ../tracking
sh run_tracking.sh "${videos_list[@]}"
cp -r output/track_results ../
echo "---------------------------------------------------"
cd ../vehicle_counting
sh run_counting.sh "${videos_list[@]}"
cp -r vehicle_counting_results ../
cd ..
echo "---------------------------------------------------"
endtime=`date +%s`
echo "TOTAL TIME : `expr $endtime - $starttime`"
exit

