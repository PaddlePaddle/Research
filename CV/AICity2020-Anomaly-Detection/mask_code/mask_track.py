#!coding: utf-8
import numpy as np
import cv2 
import os
from skimage.measure import label
from scipy.ndimage.filters import gaussian_filter
import pickle
import pdb
from numpy import zeros, ones,empty
import sys
eps = 1.0E-7

import numpy as np
import math
import sys

video_name = sys.argv[1]
f_track = "../intermediate_result/mask/abnormal_track_results/track_results_with_mask/" + video_name + ".txt"

frame_rate = 10
res_track = open(f_track,"r").readlines()

dict_track = {}
for line in res_track:
    line = line.strip().split(" ")
    frame = int(line[0])*frame_rate/30
    car_id = line[1]
    x1 = float(line[2])
    y1 = float(line[3])
    x2 = float(line[4])
    y2 = float(line[5])
    center_x = int(x1+(x2-x1)/2)
    center_y = int(y1+(y2-y1)/2)
    if not dict_track.has_key(car_id):
        dict_track[car_id] = []
        dict_track[car_id].append([frame,center_x,center_y,[int(x1),int(y1),int(x2),int(y2)]])
    else:
        dict_track[car_id].append([frame,center_x,center_y,[int(x1),int(y1),int(x2),int(y2)]])
        
        
lines = []
index_car = {}
im = cv2.imread("../intermediate_result/data/data_ori/test_data/%s/%s_00001.jpg"%(video_name,video_name),0)
h,w = im.shape
mat = np.zeros((h,w))
for car_id in dict_track.keys():
    if len(dict_track[car_id]) < 5:
        continue
    total_num = len(dict_track[car_id])
    P0 = [dict_track[car_id][0][1], dict_track[car_id][0][2]]
    Pn = [dict_track[car_id][-1][1], dict_track[car_id][-1][2]]
    
    if math.sqrt((P0[0]-Pn[0])**2 + (P0[1]-Pn[1])**2) < 8 and Pn[1] < 100:
        continue
    if math.sqrt((P0[0]-Pn[0])**2 + (P0[1]-Pn[1])**2) < 50 and Pn[1] >= 100:
        continue
    
    green = (0, 255, 0) #4
    #cv2.line(im, (P0[0], P0[1]), (Pn[0], Pn[1]), green) #5
    
    for box in dict_track[car_id]:
        h = box[3][3] - box[3][1]
        w = box[3][2] - box[3][0]
        
        if w > 50 or h > 50:
            mat[box[3][1]+int(h/4.0-1):box[3][3]-int(h/4.0-1),box[3][0]+int(w/4.0-1):box[3][2]-int(w/4.0-1)] += 1
        else:
            mat[box[3][1]:box[3][3],box[3][0]:box[3][2]] += 1
        

min_area = 200

mask= mat>0 

mask = label(mask, connectivity = 1)
num = np.max(mask)

for i in range(1,int(num+1)):
    if np.sum(mask==i)<min_area:
        mask[mask==i]=0     
mask = mask>0
mask = mask.astype(float)

kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.erode(mask, kernel_e)
mask = mask.astype(np.uint8)

mask[mask==1]=255
_,contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if cv2.contourArea(contours[i]) < 3000:
        cv2.fillConvexPoly(mask, contours[i], 0)
                        

mask_png = np.zeros(mask.shape)
mask_png[mask==255] = 1
mask_png[mask==0] = 0

if not os.path.exists("./data/mask_track/"):
    os.makedirs("./data/mask_track/")
cv2.imwrite("./data/mask_track/mask_%s.png"%str(video_name),mask_png*255)



        
    
        
