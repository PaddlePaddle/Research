import cv2
import sys
import json
import os
import numpy as np
import math



def compute_iou2(rec1, rec2):
    areas1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
    areas2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
    left = max(rec1[1],rec2[1])
    right = min(rec1[3],rec2[3])
    top = max(rec1[0], rec2[0])
    bottom = min(rec1[2], rec2[2])
    w = max(0, right-left)
    h = max(0, bottom-top)
    return float(w*h)/(areas2+areas1-w*h)

def psnr1(img1, img2):
    img1 = cv2.resize(img1,(20,20))
    img2 = cv2.resize(img2,(20,20))
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)


## pred
pred_dict={}
f = open("../../intermediate_result/pixel_track/post_process/pred_v9.txt","r")

for line in f:
    video_id = line.split(" ")[0]
    x = int(line.split(" ")[4][1:-1])
    y = int(line.split(" ")[5][:-1])
    x2 = int(line.split(" ")[6][:-1])
    y2 = int(line.split(" ")[7][:-2].strip())
    if (y2-y) == 0:
        y2 = y2+1
    if (x2-x) == 0:
        x2 = x2+1    
    if not pred_dict.has_key(video_id):
        pred_dict[video_id] = []
    
    if pred_dict[video_id] != []:
        
        for index in range(0,len(pred_dict[video_id])):
            
            re = pred_dict[video_id][index]
            rec1 = [re[4],re[5],re[6],re[7]]
            rec2 = [x,y,x2,y2]
            print(video_id,rec1,rec2,compute_iou2(rec1,rec2))
            if compute_iou2(rec1,rec2) > 0.5 and (float(line.split(" ")[1]) - float(pred_dict[video_id][index][2])) < 400:
                
                pred_dict[video_id][index][2] = line.split(" ")[2]
                break
            elif index == (len(pred_dict[video_id]) - 1):
                pred_dict[video_id].append([video_id, line.split(" ")[1], line.split(" ")[2], line.split(" ")[3], x,y,x2,y2]) # id starttime end_time conf
    else:
        pred_dict[video_id].append([video_id, line.split(" ")[1], line.split(" ")[2], line.split(" ")[3], x,y,x2,y2])    

f.close()

# ori_img
print((pred_dict))
            
for i in pred_dict.keys():
    print(i)
    video_name = i
    img_root = "../../intermediate_result/data/data_ori/test_data/" + video_name + "/"
    if not os.path.exists("txt"):
        os.makedirs("txt")
    f_out = open("./txt/" + str(video_name) + '_box.txt', 'w')
    pred_dict_i = {}
    for j in range(len(pred_dict[i])):
        
        start_time = pred_dict[i][j][1]
        end_time = pred_dict[i][j][2]
        print(start_time,end_time)
        conf = pred_dict[i][j][3]
        x = int(float(pred_dict[i][j][4]))
        y = int(float(pred_dict[i][j][5]))
        w = int(float(pred_dict[i][j][6]))-int(float(pred_dict[i][j][4])) +1
        h = int(float(pred_dict[i][j][7]))-int(float(pred_dict[i][j][5])) +1
        print(x,y,w,h)
        
        if (float(end_time) - float(start_time)) < 90: 
            continue
        if float(start_time) <= 25 and float(end_time) >= 740: #stop 
            continue
        elif float(start_time) > 25: # back
            template = cv2.imread(img_root + str(int(float(start_time))+1).zfill(6) + ".jpg")
            template_crop = template[y:y+h,x:x+w]  # 
            psnr_list_1 = []
            for t in range(max(1,int(float(start_time))-70),max(1,int(float(start_time))-70)+20,5):
                comprare_img = cv2.imread(img_root + str(int(t)).zfill(6) + ".jpg")
                comprare_img_crop = comprare_img[y:y+h,x:x+w]
                psnr_list_1.append(psnr1(comprare_img_crop,template_crop))
            
            psnr_mean_1 = (np.mean(psnr_list_1))
            
            print("start_time",video_name,psnr_mean_1)
            if psnr_mean_1 < 22:
                pred_dict_i[j] = [[j,x,y,x+w,y+h],start_time,end_time,conf]
                continue
            if w <= 5 and h <= 5:
                pred_dict_i[j] = [[j,x,y,x+w,y+h],start_time,end_time,conf]
                continue

        elif float(end_time) < 740: #forward
            template = cv2.imread(img_root + str(int(float(end_time))-5).zfill(6) + ".jpg")
            
            template_crop = template[y:y+h,x:x+w]  # 
            psnr_list_2 = []
            for t in range(int(float(end_time)+5),int(float(end_time))+25,5):
                comprare_img = cv2.imread(img_root + str(int(t)).zfill(6) + ".jpg")
                comprare_img_crop = comprare_img[y:y+h,x:x+w]
                psnr_list_2.append(psnr1(comprare_img_crop,template_crop))
            psnr_mean_2 = (np.mean(psnr_list_2))
            
            if psnr_mean_2 < 22:
                pred_dict_i[j] = [[j,x,y,x+w,y+h],start_time,end_time,conf]
                continue
                
            if w <= 5 and h <= 5:
                pred_dict_i[j] = [[j,x,y,x+w,y+h],start_time,end_time,conf]
                continue
        
    f_out.write(str(pred_dict_i))    
    f_out.close()         
            
            




