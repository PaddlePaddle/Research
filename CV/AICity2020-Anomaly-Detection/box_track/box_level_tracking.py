# code for box_level tracking
# -*- coding: UTF-8 -*-
import copy
import numpy as np
import cv2
import pickle
import math
import os

score_threshold = 0.5 # threshold for detection score
mask_threshold = 0.5 # threshold for abnormal mask
link_iou_threshold = 0.4 # linking IoU 1
iou_threshold = 0.1 # linking IoU 2
time_threshold = 500 # length threshold
skip_threshold = 10 # skip threshold to compensate for the detection performance
span_threshold = 3000 # Cross Tube Fusion threshold
merge_threshold = 7000 #temporal fusion threshold

KEYS = ['x1', 'y1', 'x_scale', 'y_scale']
im_w = float(800)
im_h = float(410)

full_rate = 30
rate = 30

Sam_interval = int(full_rate/rate)
print("Sam_interval", Sam_interval)

def psnr(img1, img2):
    # Compute the PSNR
    img1 = cv2.resize(img1,(10,10))
    img2 = cv2.resize(img2,(10,10))
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

def compute_IoU(box1, box2):
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


def multisimilarity_filter(ori_img_path, tube_data, psnr_threshold_absolute, psnr_threshold_relative):
    # Similarity Filtering Module
    filter_tube_data = []

    for i in range(len(tube_data)):
        start_id = tube_data[i][0]
        end_id = tube_data[i][1]
        tube_scores = tube_data[i][2]
        start_detres = tube_data[i][3]
        end_detres = tube_data[i][4]

        start_area = start_detres[2] * start_detres[3]
        end_area = end_detres[2] * end_detres[3]

        if float(start_id) > 360 and start_area > 35 :
            start_idx = int(float(start_id / rate))

            intra_roi_1 = []
            intra_roi_2 = []
            for tt in range(start_idx, min(892,start_idx + 26), 3):

                ori_img_path_start_id = ori_img_path + str(tt).zfill(6) + ".jpg"
                img_start_id = cv2.imread(ori_img_path_start_id)
                start_roi = img_start_id[int(start_detres[1]):int(start_detres[1] + start_detres[3]),
                            int(start_detres[0]):int(start_detres[0] + start_detres[2])]  #
                intra_roi_1.append(start_roi)

            # Compute PSNR of the intra_tube
            psnr_list_intra_1 = []
            for ii in range(len(intra_roi_1)):
                roi_a = intra_roi_1[ii]
                for jj in range(len(intra_roi_1)):
                    if ii != jj:
                        roi_b = intra_roi_1[jj]
                        psnr_list_intra_1.append(psnr(roi_a, roi_b))
            psnr_mean_intra_1 =  (np.mean(psnr_list_intra_1))

            if start_idx > 20:
                for tt in range(max(1, start_idx-40), (start_idx-10), 3):
                    img_first_id = cv2.imread(ori_img_path + str(int(tt)).zfill(6) + ".jpg")
                    first_roi = img_first_id[int(start_detres[1]):int(start_detres[1] + start_detres[3]),
                                int(start_detres[0]):int(start_detres[0] + start_detres[2])]
                    intra_roi_2.append(first_roi)
            else:
                for tt in range(max(1, start_idx-20), (start_idx-5), 2):
                    img_first_id = cv2.imread(ori_img_path + str(int(tt)).zfill(6) + ".jpg")
                    first_roi = img_first_id[int(start_detres[1]):int(start_detres[1] + start_detres[3]),
                                int(start_detres[0]):int(start_detres[0] + start_detres[2])]
                    intra_roi_2.append(first_roi)

            psnr_list_intra_2 = []
            for ii in range(len(intra_roi_2)):
                roi_a = intra_roi_2[ii]
                for jj in range(len(intra_roi_2)):
                    if ii != jj:
                        roi_b = intra_roi_2[jj]
                        psnr_list_intra_2.append(psnr(roi_a, roi_b))
            psnr_mean_intra_2 = (np.mean(psnr_list_intra_2))

            # Compute PSNR of the inter_tube
            psnr_list_inter = []
            for ii in range(len(intra_roi_1)):
                roi_a = intra_roi_1[ii]
                for jj in range(len(intra_roi_2)):
                    roi_b = intra_roi_2[jj]
                    psnr_list_inter.append(psnr(roi_a, roi_b))

            psnr_mean_inter = (np.mean(psnr_list_inter))

            if psnr_mean_inter > psnr_threshold_absolute:
                continue

            if psnr_mean_intra_2 - psnr_mean_inter < psnr_threshold_relative and psnr_mean_intra_1 - psnr_mean_inter < psnr_threshold_relative:
                continue

        elif float(end_id) < 8300*3  and end_area > 35:

            end_idx = int(float(end_id / rate))
            intra_roi_1 = []
            intra_roi_2 = []

            for tt in range(end_idx-21, end_idx, 3):

                ori_img_path_end_id = ori_img_path + str(tt).zfill(6) + ".jpg"
                img_end_id = cv2.imread(ori_img_path_end_id)
                end_roi = img_end_id[int(end_detres[1]):int(end_detres[1] + end_detres[3]),
                            int(end_detres[0]):int(end_detres[0] + end_detres[2])]  #
                intra_roi_1.append(end_roi)


            psnr_list_intra_1 = []
            for ii in range(len(intra_roi_1)):
                roi_a = intra_roi_1[ii]
                for jj in range(len(intra_roi_1)):
                    if ii != jj:
                        roi_b = intra_roi_1[jj]
                        psnr_list_intra_1.append(psnr(roi_a, roi_b))


            psnr_mean_intra_1 = (np.mean(psnr_list_intra_1))
            for tt in range(end_idx, min(891, end_idx+26), 3):
                img_last_id = cv2.imread(ori_img_path + str(int(tt)).zfill(6) + ".jpg")
                last_roi = img_last_id[int(end_detres[1]):int(end_detres[1] + end_detres[3]),
                            int(end_detres[0]):int(end_detres[0] + end_detres[2])]
                intra_roi_2.append(last_roi)

            psnr_list_intra_2 = []
            for ii in range(len(intra_roi_2)):
                roi_a = intra_roi_2[ii]
                for jj in range(len(intra_roi_2)):
                    if ii != jj:
                        roi_b = intra_roi_2[jj]
                        psnr_list_intra_2.append(psnr(roi_a, roi_b))
            psnr_mean_intra_2 = (np.mean(psnr_list_intra_2))

            psnr_list_inter = []
            for ii in range(len(intra_roi_1)):
                roi_a = intra_roi_1[ii]
                for jj in range(len(intra_roi_2)):
                    roi_b = intra_roi_2[jj]
                    psnr_list_inter.append(psnr(roi_a, roi_b))

            psnr_mean_inter = (np.mean(psnr_list_inter))

            if psnr_mean_inter > psnr_threshold_absolute:
                continue

            if (psnr_mean_intra_2 - psnr_mean_inter < psnr_threshold_relative) and \
                    (psnr_mean_intra_1 - psnr_mean_inter < psnr_threshold_relative):
                continue

        filter_tube_data.append([start_id, end_id, tube_scores, start_detres, end_detres])
    return filter_tube_data

psnr_threshold_absolute =22
psnr_threshold_relative = 2
for idx in range(1,101):
    print('searching video number %d' % (idx))

    ori_img_path = "./intermediate_result/data/data_ori/test_data/%d/" % idx

    json_dir = "./intermediate_result/box_track/video_data5_30_mask_fuse_5/" + str(idx) + ".pkl"
    with open(json_dir, "rb") as file:
        tube_all = pickle.load(file)

    tubes = tube_all[0][0]
    tube_scores = tube_all[0][1]
    tube_data = []

    same_list = []
    remove_list = []

    for kk in range(len(tubes)):
        # Cross Tube Fusion Module
        span = int(tubes[kk][-1]["frame_id"]) - int(tubes[kk][0]["frame_id"])
        # Infinite extension mechanism
        if int(tubes[kk][0]["frame_id"]) == 0:
            span += 1000

        if  span > 1500:
            if kk not in remove_list:

                start_detres_kk = tubes[kk][0]["detection_result"][0]

                store_list = []
                store_list.append(kk)

                for jj in range(len(tubes)):
                    if kk != jj:
                        start_detres_jj = tubes[jj][0]["detection_result"][0]
                        inter_iou = compute_IoU(start_detres_jj, start_detres_kk)
                        if inter_iou > iou_threshold:
                            store_list.append(jj)
                            remove_list.append(jj)
                same_list.append(store_list)

    for i in range(len(same_list)):
        len_list = len(same_list[i])
        if len_list == 1:
            ind = same_list[i][0]
            # stop mechanism
            if tubes[ind][0]["frame_id"] <= 360 and tubes[ind][-1]["frame_id"] >= 8300*3:
                continue
            span =  tubes[ind][-1]["frame_id"] - tubes[ind][0]["frame_id"]
            if tubes[ind][0]["frame_id"] == 0:
                print(tubes[ind][-1]["frame_id"], tubes[ind][0]["frame_id"])
                span += 1800
            if span < span_threshold:
                continue
            tube_data.append([tubes[ind][0]["frame_id"], tubes[ind][-1]["frame_id"], \
                              tube_scores[ind], tubes[ind][0]["detection_result"][0],
                              tubes[ind][-1]["detection_result"][0]])
        else:
            sum_tube_scores = 0
            min_start = 1000000
            max_end = 0
            for ind in same_list[i]:
                start = tubes[ind][0]["frame_id"]
                if start < min_start:
                    min_start = start
                    start_detres = tubes[ind][0]["detection_result"][0]

                end = tubes[ind][-1]["frame_id"]
                if end > max_end:
                    max_end = end
                    end_detres = tubes[ind][-1]["detection_result"][0]

                sum_tube_scores += tube_scores[ind]
            mean_tube_scores = sum_tube_scores / float(len_list)

            if min_start <= 360 and max_end >= 8300*3:
                continue
            span =  max_end - min_start
            if span < span_threshold:
                continue
            tube_data.append([min_start, max_end, mean_tube_scores, start_detres, end_detres])

    # Similarity Filtering
    filter_tube_data = multisimilarity_filter(ori_img_path, tube_data, psnr_threshold_absolute, psnr_threshold_relative)
    sort_predicton = sorted(filter_tube_data, key=lambda prediction: float(prediction[0]), reverse=False)

    # Temporal Fusion
    final = []
    if len(sort_predicton) > 1:
        start = sort_predicton[0][0]
        end = sort_predicton[0][1]
        score = sort_predicton[0][2]
        start_detres = sort_predicton[0][3]
        count = 1

        for i in range(len(sort_predicton)):
            if i > 0:
                if sort_predicton[i][0] - sort_predicton[i - 1][1] < merge_threshold:
                    end = sort_predicton[i][1]
                    count += 1
                    score = (score + sort_predicton[i][2]) / count
                else:
                    if start <= 360 and end >= 8300*3:
                        continue
                    final.append([start, end, score, start_detres])
                    start = sort_predicton[i][0]
                    end = sort_predicton[i][1]
                    score = sort_predicton[i][2]
                    start_detres = sort_predicton[i][3]
                    count = 1

            if i == len(sort_predicton) - 1:
                # stop mechanism
                if start <= 360 and end >= 8300*3:
                    continue
                final.append([start, end, score, start_detres])

    elif len(sort_predicton) == 1:
        start = sort_predicton[0][0]
        end = sort_predicton[0][1]
        score = sort_predicton[0][2]
        start_detres = sort_predicton[0][3]

        # stop mechanism
        if start <= 360 and end >= 8300*3:
            final = []
        else:
            final.append([start, end, score, start_detres])

    #print(idx, final)
    save_dir = "./intermediate_result/box_track/box_track_det"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    submission_path = './intermediate_result/box_track/box_track_det/%d.txt'%idx
    file_write_obj = open(submission_path, 'w')
    for var in final:
        anomaly_start = str((var[0]/float(rate)))
        anomaly_end = str((var[1] / float(rate)))
        anomaly_confidence = str(var[2])
        anomaly_det = str(var[3])
        line =  anomaly_start + " " + anomaly_end + " " + anomaly_confidence + " " + anomaly_det
        file_write_obj.writelines(line)
        file_write_obj.write('\n')
    file_write_obj.close()
