import numpy as np
import os
import cv2
import time

import utils_func # defined functions

# Input directory
img_dir = '../../intermediate_result/data/data_ori/test_data/' # frames dir extracted from videos
bbox_path = '../../intermediate_result/det_result/result_10frame_detmodel2_5w/' # detection result dir
mask_dir = '../../intermediate_result/mask/mask-diff/' # mask dir
# Output directory
result_dir = '../../intermediate_result/pixel_track/coarse_ddet/'

# Thresholds
frame_rate = 10
len_time_thre = 40       # minimum abnormal duration (seconds)
suspiciou_time_thre = 20 # minimum suspiciou abnormal duration (seconds)
detect_thred = 3         # the normal-suspicious state transition threshold
no_detect_thred = 3      # the suspicious/abnormal-normal state transition threshold
anomely_score_thred = 0.7 # anomely score threshold
bbox_thres = 0.7         # Detection confidence threshold
traceback_thres = 400    # backtrack time threshold
iou_thres = 0.1          # iou score threshold
raio_thres = 0.6         # relaxed constraint satisfaction ratio


# Traverse all videos
for video_id in range(1, 101):
    print('video_id: ', video_id)
    video_name = str(video_id) + '/'

    # Output dir
    if not os.path.exists(result_dir + str(video_id) + "/"):
        os.makedirs(result_dir + str(video_id) + "/")
    f_pre_out = open(res_dir_path + "pre_" + str(video_id) + '.txt', 'w')

    # Get image list
    img_names = os.listdir(os.path.join(img_dir, video_name))
    img_names.sort()
    im = cv2.imread(os.path.join(img_dir, video_name, img_names[0].split(".npy")[0]))
    h, w, c = im.shape

    # Read masks
    try:
        mask = cv2.imread(mask_dir + "mask_" + str(video_id).zfill(3) + '.jpg', 0)
        mask[mask > 0] = 1
        has_mask = True
    except:
        has_mask = False
    print('has_mask: ', has_mask)

    # -- Read detection results frame by frame --
    dt_results_fbf = {}
    for img_name in img_names:
        npy_name = img_name + '.npy'
        im_info = np.load(os.path.join(bbox_path, str(video_id) + "/" + npy_name), allow_pickle=True)
        bbox_num = len(im_info)
        if img_name not in dt_results_fbf:
            dt_results_fbf[img_name] = []
        for j in range(0, bbox_num):
            bbox_ = im_info[j][0]
            score_ = im_info[j][1]
            if score_ > bbox_thres:
                dt_results_fbf[img_name].append([int(float(bbox_[0])), int(float(bbox_[1])),
                                                 int(float(bbox_[0] + bbox_[2])),
                                                 int(float(bbox_[1] + bbox_[3])),
                                                 float(score_)]) # [x1, y1, x2, y2, score]

    # -- Define info six spatial-temporal information matrices --
    detect_count_matrix = np.zeros((h, w))
    no_detect_count_matrix = np.zeros((h, w))
    start_time_matrix = np.zeros((h, w), dtype=int)
    end_time_matrix = np.zeros((h, w), dtype=int)
    state_matrix = np.zeros((h, w), dtype=int)
    score_matrix = np.zeros((h, w))

    # Init anomaly
    all_results = []
    anomely_now = {}
    start = 0
    tmp_start = 0
    # --- Traverse all frames in a video ---
    for img_name in img_names:
        img_name = img_name.split(".npy")[0]
        t = time.time()
        img_path = os.path.join(img_dir, video_name, img_name)
        frame = cv2.imread(img_path)

        # -*- -*- spatial matrices -*- -*-
        # 1. Init spatial matrices by detection result
        tmp_detect = np.zeros((h, w)) # get ddet state for each pixel
        tmp_score = np.zeros((h, w))  # get top ddet score for each pixel
        for box in dt_results_fbf[img_name]:
            if box[3] - box[1] > 0 and box[2] - box[0] > 0:
                tmp_detect[box[1]:box[3], box[0]:box[2]] = 1
                tmp_score[int(float(box[1])):int(float(box[3])), int(float(box[0])):int(float(box[2]))] \
                    = np.maximum(box[4], tmp_score[int(float(box[1])):int(float(box[3])),
                                         int(float(box[0])):int(float(box[2]))])
        # 2. Mask filtering
        tmp_no_detect = 1 - tmp_detect
        if has_mask == True:
            tmp_detect    = tmp_detect    * mask
            tmp_no_detect = tmp_no_detect * mask
            tmp_score     = tmp_score     * mask
        # 3. Update spatial matrices
        detect_count_matrix += tmp_detect
        no_detect_count_matrix += tmp_no_detect
        no_detect_count_matrix[tmp_detect > 0] = 0 # if detected, clear no_detect_count
        score_matrix += tmp_score

        # -*- -*- temporal matrices -*- -*-
        # 4. Update temporal matrices
        start_time_matrix[detect_count_matrix == 1] = int(str(img_name).split('.jpg')[0].split('_')[2])
        end_time_matrix[detect_count_matrix > 0] = int(str(img_name).split('.jpg')[0].split('_')[2])
        state_matrix[detect_count_matrix > detect_thred] = 1
        # 5. Detect anomaly by temporal matrices
        time_delay = end_time_matrix - start_time_matrix
        time_delay = time_delay * state_matrix
        index = np.unravel_index(time_delay.argmax(), time_delay.shape) # find

        # -*- -*- anomaly judgement -*- -*-
        # 1. normality to anomaly transition
        if np.max(time_delay) / frame_rate > len_time_thre and start == 0:
            # find a longest-delay box
            time_frame = start_time_matrix[index]
            G = detect_count_matrix.copy()
            G[G < detect_count_matrix[index] - 2] = 0
            G[G > 0] = 1
            region = utils_func.search_region(G, index) # connected region

            max_iou = 0 # IOU between anomaly and boxes in frames previously
            count = 1   # relaxed satisfaction num
            tmp_len = 1 # frame num
            raio = 0.0  # relaxed satisfaction ratio
            anomely_now['region'] = region
            start_time = time_frame
            # backtrack the start time (Algorithm 2 Pixel-level Backtrack methods.)
            while (max_iou > iou_thres or tmp_len < traceback_thres or raio > raio_thres) and time_frame >= 3:
                raio = float(count) / float(tmp_len)
                max_iou = 0
                similarity_PSNR = 0  # PSNR similarity
                similarity_color = 0 # color histogram similarity
                key_time_frame = 'test_'+str(video_id)+'_'+str(time_frame).zfill(5)+'.jpg' # previous frames
                if key_time_frame in dt_results_fbf:
                    for box_i in dt_results_fbf[key_time_frame]:
                        # overlapped
                        max_iou = max(max_iou, utils_func.compute_iou(anomely_now['region'], np.array(box_i)))
                        # non-overlapped
                        try:
                            # box in previous frames
                            prev_im = cv2.imread(img_dir + str(video_id) + "/" + key_time_frame)
                            prev_bbox = prev_im[box_i[1]:box_i[3], box_i[0]:box_i[2], :]

                            # box(connected region) in anomaly
                            now_im = cv2.imread(img_dir + str(video_id) + "/test_" + str(video_id) + "_" + \
                                                str(int(anomely_now['start_time'] * frame_rate / 3) * 3 + 3).zfill(5) + ".jpg")
                            now_bbox = now_im[anomely_now['region'][1]:anomely_now['region'][3], \
                                       anomely_now['region'][0]:anomely_now['region'][2], :]
                            # restrictions for eliminating disturbances (shape, position)
                            if prev_bbox.shape[0] * prev_bbox.shape[1] < 100 \
                                    or (box_i[1]-anomely_now['region'][1]) > 3*max(prev_bbox.shape[0], now_bbox.shape[0]) \
                                    or (box_i[0]-anomely_now['region'][0]) > 3*max(prev_bbox.shape[1], now_bbox.shape[1]) \
                                    or (abs(prev_bbox.shape[0] - now_bbox.shape[0]) / float(prev_bbox.shape[0]) > 0.1 \
                                        and abs(prev_bbox.shape[1] - now_bbox.shape[1]) / float(prev_bbox.shape[1]) > 0.1):
                                pass
                            else: # calculate similarities
                                similarity_PSNR = max(similarity_PSNR, utils_func.psnr(now_bbox, prev_bbox))
                                similarity_color = max(similarity_color, utils_func.compare_similar_hist( \
                                                                             utils_func.calc_bgr_hist(now_bbox), \
                                                                             utils_func.calc_bgr_hist(prev_bbox)))
                        except:
                            continue
                # relaxed constraints to deal with discontinuous detection results
                # relaxed threshold
                if max_iou > 0.3 or similarity_PSNR > 18 or similarity_color > 0.88:
                    count += 1
                    # update threshold
                    if max_iou > 0.5 or similarity_PSNR > 20 or similarity_color > 0.9:
                        start_time = time_frame
                time_frame -= 3
                tmp_len += 1
            time_frame = start_time
            # update the start/end time
            anomely_now['start_time'] = max(0, start_time / frame_rate)
            anomely_now['end_time']   = max(0, end_time_matrix[index] / frame_rate)
            start = 1

        # 2. normality to suspicious anomaly transition
        elif np.max(time_delay) / frame_rate > suspiciou_time_thre and tmp_start == 0:
            time_frame = int(start_time_matrix[index])
            G = detect_count_matrix.copy()
            G[G < detect_count_matrix[index] - 2] = 0
            G[G > 0] = 1
            region = utils_func.search_region(G, index)

            anomely_now['region'] = region
            # (No backtrack process here compared with normality to anomaly transition)
            anomely_now['start_time'] = max(0, time_frame / frame_rate)
            anomely_now['end_time']   = max(0, end_time_matrix[index] / frame_rate)
            tmp_start = 1

        # 3. anomaly to normality transition
        if np.max(time_delay) / frame_rate > len_time_thre and start == 1:
            if no_detect_count_matrix[index] > no_detect_thred:
                anomely_score = score_matrix[index] / detect_count_matrix[index]
                if anomely_score > anomely_score_thred:
                    anomely_now['end_time'] = end_time_matrix[index] / frame_rate
                    anomely_now['score'] = anomely_score
                    # record candidates and clear anomely_now
                    all_results.append(anomely_now)
                    anomely_now = {}
                start = 0

        # 4. suspicious anomaly to normality transition
        elif np.max(time_delay) / frame_rate > suspiciou_time_thre and tmp_start == 1:
            if no_detect_count_matrix[index] > no_detect_thred:
                anomely_score = score_matrix[index] / detect_count_matrix[index]
                if anomely_score > anomely_score_thred:
                    anomely_now['end_time'] = end_time_matrix[index] / frame_rate
                    anomely_now['score'] = anomely_score
                    # (no record candidates here)
                tmp_start = 0

        # update matrices
        state_matrix[no_detect_count_matrix > no_detect_thred] = 0
        no_detect_count_matrix[no_detect_count_matrix > no_detect_thred] = 0
        tmp_detect = tmp_detect + state_matrix
        tmp_detect[tmp_detect > 1] = 1
        detect_count_matrix = detect_count_matrix * tmp_detect
        score_matrix = score_matrix * tmp_detect

    # record candidates and clear anomely_now
    if np.max(time_delay) / frame_rate > len_time_thre and start == 1:
        anomely_score = score_matrix[index] / detect_count_matrix[index]
        if anomely_score > anomely_score_thred:
            anomely_now['end_time'] = end_time_matrix[index] / frame_rate
            anomely_now['score'] = anomely_score
            all_results.append(anomely_now)
            anomely_now = {}
            start = 0

    # --- sort and write results ---
    last_time = []
    for item in all_results:
        last_time.append([item["start_time"], item["end_time"], item["region"]])
    last_time.sort()
    if last_time:
        for i in range(len(last_time)):
            # write (starttime, endtime, bbox) to txt
            f_pre_out.write(str(last_time[i][0]) + " " + \
                            str(last_time[i][1]) + " " + \
                            str(last_time[i][2]) + "\n")
    f_pre_out.close()

