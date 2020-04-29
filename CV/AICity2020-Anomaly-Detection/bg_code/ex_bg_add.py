# -*- coding: utf-8 -*-
import cv2, os

rt = '../intermediate_result/data/AIC20_track4/test-data/'
out_dir = '../intermediate_result/bg/'
frame_rate = 30

# Background modeling in forward direction
for i in range(1, 101):
    id = str(i)
    # path for background frames
    wrt_bg = out_dir + 'bg_add_fps30_forward/' + id + '/'
    if not os.path.exists(wrt_bg):
        os.makedirs(wrt_bg)

    if os.path.exists(os.path.join(rt, id+'.mp4')):
        cap = cv2.VideoCapture(os.path.join(rt, id+'.mp4'))

    ret, frame = cap.read()

    bg_img = frame
    count = 0
    alpha = 0.05
    while ret:
        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, 1.0/frame_rate * 1000 * count)
        ret, frame = cap.read()
        if ret == False:
            continue

        bg_img = (bg_img*(1-alpha) + frame*alpha).astype('uint8')
        cv2.imwrite(os.path.join(wrt_bg, 'test_'+id+'_'+str(int(count)).zfill(5)+'.jpg'), bg_img)
