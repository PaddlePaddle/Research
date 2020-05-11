# construct tube for box_level tracking

# -*- coding: UTF-8 -*-

import json
import copy
import numpy as np
import cv2
from multiprocessing import Pool, current_process
import pickle
import os

score_threshold = 0.5 # 对于检测分数选择阈值
mask_threshold = 0.5 # 对于道路主体mask的阈值
link_iou_threshold = 0.4 # 链接起来形成tube的阈值
iou_threshold = 0.8 #停止的车辆至少有多少的iou交集
iou_threshold_merge = 0.1 #停止的车辆至少有多少的iou交集
time_threshold = 100 #跨过多少帧就进入候选考虑范围
skip_threshold = 10 #允许跳过几个帧，弥补检测性能的缺失
span_threshold = 1000 #扩散的阈值，50s
merge_threshold = 3500 #前一个结束与后一个开始的允许阈值
stop_freqency_threshold = 0.5
# for idx in range(100):
KEYS = ['x1', 'y1', 'x_scale', 'y_scale']
im_w = float(800)
im_h = float(410)

worker_cnt = 10
full_rate = 30
rate = 30
Sam_interval = int(full_rate/rate)
print("Sam_interval", Sam_interval)

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

def get_top_tubes(det_list_org, link_iou_threshold, skip_threshold, time_threshold):
    # tube construction process
    det_list = copy.deepcopy(det_list_org)
    tubes = []
    tube_scores = []
    continue_flg = True

    while continue_flg:
        # find the top score in the current det_list
        max_score = 0
        max_timestep = 0
        max_boxid = 0
        for i in range(len(det_list)):
            for j in range(len(det_list[i]["detection_result"])):
                if det_list[i]["detection_result"][j][-1] > max_score:
                    max_score = det_list[i]["detection_result"][j][-1]
                    max_timestep = i
                    max_boxid = j

        tube_id = []
        tube_id.append((max_timestep, max_boxid, 1))

        # forward
        timestep = max_timestep
        continue_flag = True
        while timestep + 1 < len(det_list) and continue_flag == True:
            curbox_coods = det_list[tube_id[-1][0]]["detection_result"][tube_id[-1][1]][0]
            n_curbox = len(det_list[timestep + 1]["detection_result"])
            add_flag = False
            if n_curbox > 0:
                max_link_score = 0
                for i_nextbox in range(n_curbox):
                    nextbox_coods = det_list[timestep + 1]["detection_result"][i_nextbox][0]
                    link_score = compute_IoU(nextbox_coods, curbox_coods)

                    if link_score > link_iou_threshold and link_score > max_link_score:
                        max_boxid = i_nextbox
                        max_link_score = link_score
                        add_flag = True

            if timestep + 1 - tube_id[-1][0] > skip_threshold:
                continue_flag = False
            elif add_flag == True:
                tube_id.append((timestep + 1, max_boxid, max_link_score))

            timestep += 1

            if timestep + 1 == len(det_list):
                continue_flag = False

        # backward
        timestep = max_timestep
        continue_flag = True
        while timestep - 1 >= 0 and continue_flag == True:
            curbox_coods = det_list[tube_id[0][0]]["detection_result"][tube_id[0][1]][0]
            n_curbox = len(det_list[timestep - 1]["detection_result"])
            add_flag = False
            if n_curbox > 0:
                max_link_score = 0
                for i_prevbox in range(n_curbox):
                    prevbox_coods = det_list[timestep - 1]["detection_result"][i_prevbox][0]
                    link_score = compute_IoU(prevbox_coods, curbox_coods)

                    if link_score > iou_threshold and link_score > max_link_score:
                        max_boxid = i_prevbox
                        max_link_score = link_score
                        add_flag = True

            if tube_id[0][0] - timestep - 1 > skip_threshold:
                continue_flag = False
            elif add_flag == True:
                tube_id.insert(0, (timestep - 1, max_boxid, max_link_score))

            timestep -= 1

            if timestep <= 0:
                continue_flag = False

        # get path and remove used boxes
        cur_tube = []
        tube_score = 0
        for i, tube_path in enumerate(tube_id):

            inform = {}
            inform["frame_id"] = tube_path[0]
            inform["detection_result"] = det_list[tube_path[0]]["detection_result"][tube_path[1]]
            inform["link_score"] = tube_path[2]
            tube_score += tube_path[2]

            cur_tube.append(inform)
            det_list[tube_path[0]]["detection_result"] = np.delete(det_list[tube_path[0]]["detection_result"],
                                                                   tube_path[1], 0)

            sum_detection = 0
            for kk in range(len(det_list)):
                sum_detection += len(det_list[kk]["detection_result"])
            if sum_detection == 0:
                continue_flg = False

        if len(tube_id) > time_threshold:
            tube_score = tube_score / len(tube_id)
            tubes.append(cur_tube)
            tube_scores.append(tube_score)

    return tubes, tube_scores


def process(idx):
    print('searching video number %s' % (idx))

    idx = int(idx)
    video_data = []
    mask_path = "./intermediate_result/mask/mask-fuse/mask_%03d.jpg" %idx
    mask_img = cv2.imread(mask_path)
    imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    thresh = 128
    ret, binary_mask = cv2.threshold(imgray, thresh, 1, cv2.THRESH_BINARY)

    for num in range(1,30000):
        try:
            current_id = num * Sam_interval
            Det_path = "./intermediate_result/det_result/result_30frame_detmodel2_5w/%d/test_%d_%05d.jpg.npy" % (idx, idx, current_id)
            det_res = np.load(Det_path, allow_pickle=True)
            inform = {}
            inform["image_id"] = num
            len_box = len(det_res)
            det_filter = []
            for i in range(len_box):
                if det_res[i][2] > score_threshold:
                    x1 = int(det_res[i][0][0])
                    y1 = int(det_res[i][0][1])
                    x2 = int(det_res[i][0][0] + det_res[i][0][2])
                    y2 = int(det_res[i][0][1] + det_res[i][0][3])
                    mask_area = (binary_mask[y1:y2, x1:x2]).sum()
                    area = (y2 - y1) * (x2 - x1)
                    mask_iou = mask_area / float(area)
                    if mask_iou >= mask_threshold:
                        det_filter.append((det_res[i][0], det_res[i][1], det_res[i][2]))
            inform["detection_result"] = det_filter
            video_data.append(inform)
        except:
            pass


    tube_all = []
    tubes, tube_scores = get_top_tubes(video_data,  link_iou_threshold, skip_threshold, time_threshold)
    tube_all.append((tubes, tube_scores))

    save_dir = "./intermediate_result/box_track/video_data5_30_mask_fuse_5"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    json_dir = "./intermediate_result/box_track/video_data5_30_mask_fuse_5/" + str(idx) + ".pkl"

    with open(json_dir, "wb") as file:
        pickle.dump(tube_all, file)


threads = []

idx_list = []
happen_list = []

for idx in range(1,101):
    idx_list.append(str(idx))

pool = Pool(processes=worker_cnt)
pool.map(process, idx_list)

