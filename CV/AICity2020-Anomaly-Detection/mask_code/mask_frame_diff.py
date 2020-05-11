import cv2
import os
import numpy as np
import sys
import skimage
from skimage.measure import label 
from scipy.ndimage.filters import gaussian_filter

rt = '../intermediate_result/data/AIC20_track4/test-data/'
videos = os.listdir(rt)

wrt_fgmask = './data/mask_diff/' 


if not os.path.exists(wrt_fgmask):
    os.makedirs(wrt_fgmask)

video = sys.argv[1]
if True:
    count = 0
    print (video)
    out = 0 

    #read video
    cap = cv2.VideoCapture(os.path.join(rt, video + ".mp4"))
    ret, frame = cap.read()
    
    while ret:
        if count % 5 == 0:
            last_frame = frame
        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, 0.2 * 1000 * count)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        fg = cv2.subtract(frame,last_frame)
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, fg1 = cv2.threshold(fg, 100, 255, cv2.THRESH_BINARY)
        fg1[fg1==255] = 1
        
        if sum(sum(fg1)) > 13000:
            continue
        
        out = cv2.bitwise_or(out,fg) #||
        
        out = cv2.medianBlur(out, 3) 
        
        out = cv2.GaussianBlur(out, (3, 3), 0) 
        
        _, out = cv2.threshold(out, 99, 255, cv2.THRESH_BINARY)
        
        
    min_area = 10000   
    mask = label(out, connectivity = 1)
    num = np.max(mask)
    for i in range(1,int(num+1)):
        if np.sum(mask==i)<min_area:
            mask[mask==i]=0     
    mask = mask>0
    mask = mask.astype(float)

    cv2.imwrite(os.path.join(wrt_fgmask, str(int(video.split('.')[0])).zfill(3) + '.jpg'), mask*255)


