import math
import os
import sys
import cv2
import numpy as np
from PIL import Image,ImageStat
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

def brightness(im_file,x1,y1,x2,y2):
    im = Image.open(im_file)
    box = (x1,y1,x2,y2)
    im = im.crop(box)
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def psnr1(img1, img2):
    img1 = cv2.resize(img1,(20,20))
    img2 = cv2.resize(img2,(20,20))
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)


for video_name in range(1,101):
    try:
        line = open("./txt/"+ str(video_name) + "_box.txt","r")
    except:
        continue
    f_out = open("./txt/" + str(video_name) + '_time_back_box.txt', 'w')
    
    print(video_name)
    video_result = eval(line.read())
    print(video_result)
    for key in video_result:
        start_time = float(video_result[key][1])
        if start_time < 5:
            video_result[key][1] = 0
            continue
        x1,y1,x2,y2 = video_result[key][0][1],video_result[key][0][2],video_result[key][0][3],video_result[key][0][4]
        img_root = "../../intermediate_result/data/data_ori/test_data/" + str(video_name) + "/"

        template = cv2.imread(img_root + str(int(float(start_time))+1).zfill(6) + ".jpg")
        template_crop = template[y1:y2,x1:x2]

        for t in range(int(float(start_time)),5,-1):
            psnr1_list = []
            psnr1_ = []
            comprare_img = cv2.imread(img_root + str(int(t)).zfill(6) + ".jpg")
            comprare_img_crop = comprare_img[y1:y2,x1:x2]
            psnr1_list.append(psnr1(comprare_img_crop,template_crop))
            psnr1_.append(psnr1(comprare_img_crop,template_crop)<19)
            comprare_img = cv2.imread(img_root + str(int(t-1)).zfill(6) + ".jpg")
            comprare_img_crop = comprare_img[y1:y2,x1:x2]
            psnr1_list.append(psnr1(comprare_img_crop,template_crop))
            psnr1_.append(psnr1(comprare_img_crop,template_crop)<19)
            comprare_img = cv2.imread(img_root + str(int(t-2)).zfill(6) + ".jpg")
            comprare_img_crop = comprare_img[y1:y2,x1:x2]
            psnr1_list.append(psnr1(comprare_img_crop,template_crop))
            psnr1_.append(psnr1(comprare_img_crop,template_crop)<19)
            comprare_img = cv2.imread(img_root + str(int(t-3)).zfill(6) + ".jpg")
            comprare_img_crop = comprare_img[y1:y2,x1:x2]
            psnr1_list.append(psnr1(comprare_img_crop,template_crop))
            psnr1_.append(psnr1(comprare_img_crop,template_crop)<19)
            comprare_img = cv2.imread(img_root + str(int(t-4)).zfill(6) + ".jpg")
            comprare_img_crop = comprare_img[y1:y2,x1:x2]
            psnr1_list.append(psnr1(comprare_img_crop,template_crop))
            psnr1_.append(psnr1(comprare_img_crop,template_crop)<19)
            print(np.sum(psnr1_))
            print(psnr1_list)
            #if np.mean(psnr1_list) < 21:
            img_file = img_root + str(int(t)).zfill(6) + ".jpg"
            bt = brightness(img_file,x1,y1,x2,y2)
            bt_all = brightness(img_file,0,0,799,409)

            if (bt - bt_all) < 100 and np.sum(psnr1_) > 4:
                video_result[key][1] = str(t)
                break
            if (bt - bt_all) >= 100 and float(np.mean(psnr1_list)) <= 15:
                video_result[key][1] = str(t)
                break
            
    print(video_result)            
    line.close()
    f_out.write(str(video_result)) 
    f_out.close()   
        
        
    
    
