import sys
import os
import cv2
import numpy as np 
import math


video_name = sys.argv[1]


line = open("./txt/"+ str(video_name) + "_time_back_box.txt","r")
print(line)
video_result = eval(line.read())
last_time = []
for key in video_result:
    last_time.append([float(video_result[key][1]), float(video_result[key][2]), video_result[key][0]])
last_time.sort()
print(last_time)
result_ = []
last_id = 0
if last_time:
    endding = 0
    
    for id in range(len(last_time)):
        print(last_time[id])
        
        if last_time[id][0] - endding > 120 or id == 0:
            result_.append(last_time[id])
            last_id = id 
            # update max_lowerThanStart
        elif id > 0:
            result_[last_id][1] = last_time[id][1]
            
        if id == len(last_time) - 1:
            break
        if last_time[id][1] > endding:
            endding = last_time[id][1]
img_root = "../../intermediate_result/data/data_ori/test_data/" + video_name + "/"
final_box = {}
if result_:
    f_out = open("./txt/" + str(video_name) + '.txt', 'w')
    f_out_box = open("./txt/" + str(video_name) + '_final_box.txt', 'w')
    for i in range(len(result_)):
        start_time = result_[i][0]
        end_time = result_[i][1]
        conf = 1
        x = int(float(result_[i][2][1]))
        y = int(float(result_[i][2][2]))
        w = int(float(result_[i][2][3]))-int(float(result_[i][2][1]))
        h = int(float(result_[i][2][4]))-int(float(result_[i][2][2]))
        f_out.write(str(start_time) + " " + str(conf))
        f_out.write("\n")
        final_box[i] = [[i,x,y,x+w,y+h],start_time,end_time,conf]
                  
            
    f_out_box.write(str(final_box))
    f_out_box.close()
    f_out.close()         
            
            




