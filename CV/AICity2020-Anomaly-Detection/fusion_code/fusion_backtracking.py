# code for fusion and backtracking optimization

# -*- coding: UTF-8 -*-
import numpy as np
import math
import os
import cv2

score_threshold = 0.5
mask_threshold = 0.5
max_traceback_frame = 170
traceback_IoU_threshold = 0.6
back_skip_threshold = 21
merge_threshold = 100

eps = 1e-8

KEYS = ['x1', 'y1', 'x_scale', 'y_scale']
im_w = float(800)
im_h = float(410)

def compute_IoU(box1, box2):
    # Compute the IoU between two boxes
    if isinstance(box1, list):
        box1 = {key: val for key, val in zip(KEYS, box1)}
    if isinstance(box2, list):
        box2 = {key: val for key, val in zip(KEYS, box2)}

    box1['x2'] = box1['x1'] + box1['x_scale']
    box1['y2'] = box1['y1'] + box1['y_scale']

    box2['x2'] = box2['x1'] + box2['x_scale']
    box2['y2'] = box2['y1'] + box2['y_scale']
    width = max(min(box1['x2'] / im_w, box2['x2'] / im_w) - max(box1['x1'] / im_w, box2['x1'] / im_w), 0)
    height = max(min(box1['y2'] / im_h, box2['y2'] / im_h) - max(box1['y1'] / im_h, box2['y1'] / im_h), 0)
    intersection = width * height
    box1_area = (box1['x2'] / im_w - box1['x1'] / im_w) * (box1['y2'] / im_h - box1['y1'] / im_h)
    box2_area = (box2['x2'] / im_w - box2['x1'] / im_w) * (box2['y2'] / im_h - box2['y1'] / im_h)

    union = box1_area + box2_area - intersection

    return float(intersection) / float(union)

for video_name in range(1,101):
    video_name = int(video_name)

    # load the results of pixel_level tracking branch
    pixel_final = []
    try:
        result_txt = "./intermediate_result/pixel_track/post_process/txt/%d_time_back_box.txt" % video_name

        f_pre = open(result_txt, 'r')
        video_result = eval(f_pre.read())
        prediction = []
        for key in video_result:
            start_time = float(video_result[key][1])
            end_time = float(video_result[key][2])
            x1, y1, x2, y2 = video_result[key][0][1], video_result[key][0][2], video_result[key][0][3], \
                             video_result[key][0][4]

            start_des = [x1 ,y1, x2-x1, y2-y1]
            prediction.append([start_time, end_time, 1.0, start_des])

        sort_predicton = sorted(prediction, key=lambda prediction: float(prediction[0]), reverse=False)

        if len(sort_predicton) > 1:
            start = sort_predicton[0][0]
            end = sort_predicton[0][1]
            score = sort_predicton[0][2]
            start_detres = sort_predicton[0][3]
            count = 1

            for i in range(len(sort_predicton)):
                if i > 0:
                    if sort_predicton[i][0] - sort_predicton[i - 1][1] < merge_threshold:
                        end = max(sort_predicton[i-1][1], sort_predicton[i][1])
                        count += 1
                        score = (score + sort_predicton[i][2]) / count
                    else:
                        if start <= 12 and end >= 830:
                            continue
                        pixel_final.append([start, end, score, start_detres])
                        start = sort_predicton[i][0]
                        end = sort_predicton[i][1]
                        score = sort_predicton[i][2]
                        start_detres = sort_predicton[i][3]
                        count = 1

                if i == len(sort_predicton) - 1:
                    # stop mechanism
                    if start <= 12 and end >= 830:
                        continue
                    pixel_final.append([start, end, score, start_detres])

        elif len(sort_predicton) == 1:
            start = sort_predicton[0][0]
            end = sort_predicton[0][1]
            score = sort_predicton[0][2]
            start_detres = sort_predicton[0][3]

            if start <= 12 and end >= 830:
                pixel_final = []
            else:
                pixel_final.append([start, end, score, start_detres])

    except:
        pass

    # load the results of box_level tracking branch
    box_final = []
    try:
        f_pre = open('./intermediate_result/box_track/box_track_det/' + str(video_name) + '.txt', 'r')
        for line in f_pre:
            line = line.strip()
            anomaly_start = float(line.split(' ')[0])
            anomaly_end = float(line.split(' ')[1])
            anomaly_confidence = float(line.split(' ')[2])
            x = float(line.split(" ")[3][1:-1])
            y = float(line.split(" ")[4][:-1])
            w = float(line.split(" ")[5][:-1])
            h = float(line.split(" ")[6][:-2].strip())
            anomaly_det = [x,y,w,h]

            box_final.append([anomaly_start, anomaly_end, anomaly_confidence, anomaly_det])
    except:
        pass

    # Fusion
    final = []
    if len(pixel_final) == 0 and len(box_final) == 0:
        final = []
    elif len(pixel_final) > 0 and len(box_final) == 0:
        final = pixel_final
    elif len(pixel_final) == 0 and len(box_final) > 0:
        final = box_final
    else:

        len_pixel = len(pixel_final)
        len_box = len(box_final)
        min_len = min(len_pixel, len_box)

        if len_pixel >= len_box:
            for i in range(min_len):
                pixel_start_time = pixel_final[i][0]
                box_start_time = box_final[i][0]
                if pixel_start_time <= box_start_time:
                    final.append(pixel_final[i])
                else:
                    final.append(box_final[i])

            for i in range(len_pixel-len_box):
                final.append(pixel_final[i+len_box])
        else:
            for i in range(min_len):
                pixel_start_time = pixel_final[i][0]
                box_start_time = box_final[i][0]
                if pixel_start_time <= box_start_time:
                    final.append(pixel_final[i])
                else:
                    final.append(box_final[i])

            for i in range(len_box-len_pixel):
                final.append(box_final[i+len_pixel])

    # Backtracking Optimization
    for idd in range(len(final)):

        start_idx = int(np.around(final[idd][0]) * 30)
        end_idx = int(np.around(final[idd][1]) * 30)
        if start_idx < 120:
            final[idd][0] = 0

        else:
            previous_id = start_idx
            previous_detres = final[idd][-1]

            mask_path = "./intermediate_result/mask/mask-diff/mask_%03d.jpg" % video_name  # mask_fuse

            mask_img = cv2.imread(mask_path)
            imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            thresh = 128
            ret, binary_mask = cv2.threshold(imgray, thresh, 1, cv2.THRESH_BINARY)

            # back search
            continue_flag = True
            while continue_flag == True:
                for current_id in range(start_idx, 0, -3):
                    # print(current_id)
                    Det_path = "./intermediate_result/det_result/result_10frame_detmodel2_inverse/%d/test_%d_%05d.jpg.npy" % (
                        video_name, video_name, current_id)
                    det_res = np.load(Det_path, allow_pickle=True)
                    len_box = len(det_res)
                    det_filter = []
                    for i in range(len_box):
                        if det_res[i][2] > score_threshold:
                            x1 = int(det_res[i][0][0])
                            y1 = int(det_res[i][0][1])
                            x2 = int(det_res[i][0][0] + det_res[i][0][2])
                            y2 = int(det_res[i][0][1] + det_res[i][0][3])
                            # print(binary_mask[y1:y2, x1:x2])
                            mask_area = (binary_mask[y1:y2, x1:x2]).sum()
                            area = (y2 - y1) * (x2 - x1)
                            mask_iou = mask_area / float(area)

                            connect_iou = compute_IoU(previous_detres, det_res[i][0])
                            # print(mask_iou)
                            if mask_iou >= mask_threshold and connect_iou > traceback_IoU_threshold:
                                previous_detres = det_res[i][0]
                                previous_id = current_id

                    if previous_id - current_id > back_skip_threshold or current_id == 3 or start_idx  - current_id > max_traceback_frame:
                        continue_flag = False
                        # stop mechanism:
                        if current_id < 12 and end_idx > 830:
                            final = np.delete(final,
                                              idd, 0)
                        else:
                            final[idd][0] = current_id / float(30)
                            # change start time:
                            if final[idd][0] < 5:
                                final[idd][0] = 0
                        break

    save_dir = "./intermediate_result/final_result/merge"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    submission_path = './intermediate_result/final_result/merge/%d.txt' % video_name
    file_write_obj = open(submission_path, 'w')  # 以写的方式打开文件，如果文件不存在，就会自动创建
    for var in final:
        anomaly_start = str((var[0]))
        anomaly_confidence = str(var[2])
        line = anomaly_start + " " + anomaly_confidence
        file_write_obj.writelines(line)
        file_write_obj.write('\n')
    file_write_obj.close()


# form the submitted documents
pred_dict = []
for video_name in range(1,101):
    try:
        f_pre = open('./intermediate_result/final_result/merge/' + str(video_name) + '.txt','r')
        for line in f_pre:
            line = line.strip()
            anomaly_start = line.split(' ')[0]
            anomaly_confidence = line.split(' ')[1]
            pred_dict.append([video_name, anomaly_start, anomaly_confidence])
    except:
        pass

sort_predicton = sorted(pred_dict, key=lambda prediction: float(prediction[2]), reverse=True)
sort_predicton = sort_predicton[:100]

submission_path = './intermediate_result/final_result/track4.txt'
file_write_obj = open(submission_path, 'w')

sort_predicton_new = sorted(sort_predicton, key=lambda x: x[0], reverse=False)

for var in sort_predicton_new:
    video_name = str(var[0])
    anomaly_start_float = str(float(var[1]))
    anomaly_start = str(var[1])
    anomaly_confidence = str(var[2])
    line =  video_name + " " +  anomaly_start_float + " " + anomaly_confidence
    file_write_obj.writelines(line)
    file_write_obj.write('\n')
file_write_obj.close()


