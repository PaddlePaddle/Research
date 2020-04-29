# -*- coding: utf-8 -*-
import cv2, os

rt = '../intermediate_result/data/AIC20_track4/test-data/'
out_dir = '../intermediate_result/bg/'
frame_rate = 30

# Background modeling in forward direction
for i in range(1, 101):
    id = str(i)
    # path for background frames
    wrt_bg = out_dir + 'bg_mog_fps30_forward/' + id + '/'
    if not os.path.exists(wrt_bg):
        os.makedirs(wrt_bg)

    if os.path.exists(os.path.join(rt, id+'.mp4')):
        cap = cv2.VideoCapture(os.path.join(rt, id+'.mp4'))

    ret, frame = cap.read()

    # build MOG2 model
    bs = cv2.createBackgroundSubtractorMOG2(120, 16, False)

    count = 0
    while ret:

        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, 1.0/frame_rate * 1000 * count)
        ret, frame = cap.read()
        if ret == False:
            break

        fg_mask = bs.apply(frame)
        bg_img = bs.getBackgroundImage()

        cv2.imwrite(os.path.join(wrt_bg, 'test_'+id + '_' + str(int(count)).zfill(5) + '.jpg'), bg_img)


# Background modeling in backward direction
for i in range(1, 101):
    id = str(i)
    # path for background frames
    wrt_bg = out_dir + 'bg_mog_fps30_backward/' + id + '/'
    if not os.path.exists(wrt_bg):
        os.makedirs(wrt_bg)
    cap = cv2.VideoCapture(os.path.join(rt,  id + '.mp4'))

    # get video frames
    if cap.isOpened():
        rate = cap.get(5)  # frame rate
        FrameNumber = cap.get(7)  # total frame number
        duration = FrameNumber / rate   # total length

    # build MOG2 model ( inverse! )
    cap.set(cv2.CAP_PROP_POS_MSEC,  (duration - 1.0/frame_rate) * 1000)
    ret, frame = cap.read()

    bs = cv2.createBackgroundSubtractorMOG2(120, 16, False) # 120 for 30fps, T = 4s
    count = 0
    while ret:
        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, (duration - 1.0/frame_rate * count) * 1000)
        ret, frame = cap.read()
        if ret == False or int(round(duration * frame_rate - count)) == 0:
            break
        else:
            print(count,  round(duration * frame_rate - count))

        fg_mask = bs.apply(frame)
        bg_img = bs.getBackgroundImage()
        cv2.imwrite(os.path.join(wrt_bg, 'test_'+id + '_' + str(int(round(duration * frame_rate - count))).zfill(5) + '.jpg'), bg_img)


