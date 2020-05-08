import json
import os

import cv2
import numpy as np

winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
winStride = (8, 8)
padding = (8, 8)
color_hist_size = 128


def get_hog_features(img, winSize, blockSize, blockStride, cellSize, nbins):
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img = cv2.resize(img, winSize, interpolation=cv2.INTER_CUBIC)
    hog_feature = hog.compute(img, winStride, padding).reshape((-1,))
    return hog_feature


def get_bgr_features(image):
    b_hist = cv2.calcHist([image], [0], None, [color_hist_size], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [color_hist_size], [0, 256])
    r_hist = cv2.calcHist([image], [2], None, [color_hist_size], [0, 256])
    return np.concatenate((b_hist, g_hist, r_hist)).reshape((-1,))


def read_files(path):
    assert os.path.isfile(path), 'path wrong!'
    box_dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip().split(' ')
            frame_id = int(line[0])
            track_id = int(line[1])
            bbox = list(map(float, line[2:6]))
            if track_id in box_dict.keys():
                box_dict[track_id]["frame_ids"].append(frame_id)
                box_dict[track_id]["boxs"].append(bbox)
                box_dict[track_id]["features"].append([])
            else:
                box_dict[track_id] = {"frame_ids": [frame_id], "boxs": [bbox], "features": [[]]}
    return box_dict


def read_camera_config(path):
    assert os.path.isfile(path), 'path wrong!'
    with open(path) as f:
        return json.load(f)


class CrossJudger(object):
    # 判断线段与矩形相交
    def cross(self, p1, p2, p3):  # 叉积判定
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1

    def segment(self, p1, p2, p3, p4):  # 判断两线段是否相交
        # 矩形判定，以l1、l2为对角线的矩形必相交，否则两线段不相交
        if (max(p1[0], p2[0]) >= min(p3[0], p4[0])  # 矩形1最右端大于矩形2最左端
                and max(p3[0], p4[0]) >= min(p1[0], p2[0])  # 矩形2最右端大于矩形1最左端
                and max(p1[1], p2[1]) >= min(p3[1], p4[1])  # 矩形1最高端大于矩形2最低端
                and max(p3[1], p4[1]) >= min(p1[1], p2[1])):  # 矩形2最高端大于矩形1最低端
            if (self.cross(p1, p2, p3) * self.cross(p1, p2, p4) <= 0
                    and self.cross(p3, p4, p1) * self.cross(p3, p4, p2) <= 0):
                D = 1
            else:
                D = 0
        else:
            D = 0
        return D

    def judge(self, p1, p2, box):  # box=[x1, y1, x2, y2]
        # step 1 check if end point is in the square
        # 相交返回True，不相交返回False
        if (p1[0] >= box[0] and p1[0] <= box[2] and p1[1] >= box[1] and p1[1] <= box[3]) or (
                p2[0] >= box[0] and p2[0] <= box[2] and p2[1] >= box[1] and p2[1] <= box[3]):
            return True
        else:
            # step 2 check if diagonal cross the segment
            tl = [box[0], box[1]]
            br = [box[2], box[3]]
            bl = [box[0], box[3]]
            tr = [box[2], box[1]]
            if self.segment(p1, p2, tl, br) or self.segment(p1, p2, bl, tr):
                return True
            else:
                return False

    def visualize(self, pt1, pt2, box, img_shape=(960, 1280, 3), cross_flag=False):
        image = np.zeros(img_shape)
        cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255))
        x1, y1, x2, y2 = list(map(int, box))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        if cross_flag:
            cv2.rectangle(image, (480, 640), (580, 740), (255, 255, 255), thickness=8)
        cv2.imshow("test", image)
        cv2.waitKey(0)


def class_tracklet(tracklet_dict, camera_config):
    ret_dict = {}
    ret_dict["start"] = []
    ret_dict["stop"] = []
    cj = CrossJudger()

    for tid, tracklet in tracklet_dict.items():
        results_flag = []
        results_fid = []
        for m_name, value in camera_config.items():
            cross_flag_start = False
            cross_flag_stop = False
            """
            True True -> 0
            True False -> 1
            False True -> 2
            False False -> 3
            """
            start_fid, stop_fid = -1, -1
            for tlb, fid in zip(tracklet["boxs"], tracklet["frame_ids"]):
                start_flag = cj.judge(value["src"]["point_1"], value["src"]["point_2"], tlb)
                cross_flag_start = cross_flag_start or start_flag
                if start_flag:
                    start_fid = fid
                    continue
                stop_flag = cj.judge(value["dst"]["point_1"], value["dst"]["point_2"], tlb)
                # cj.visualize(value["dst"]["point_1"], value["dst"]["point_2"], tlb, cross_flag=stop_flag)
                cross_flag_stop = cross_flag_stop or stop_flag
                if stop_flag:
                    stop_fid = fid
            if cross_flag_start and cross_flag_stop:
                results_flag.append(0)
                results_fid.append([start_fid, stop_fid])
            elif cross_flag_start and not cross_flag_stop:
                results_flag.append(1)
                results_fid.append([start_fid, stop_fid])
            elif not cross_flag_start and cross_flag_stop:
                results_flag.append(2)
                results_fid.append([start_fid, stop_fid])
            else:
                results_flag.append(3)
                results_fid.append([start_fid, stop_fid])
        if 0 in results_flag:
            continue
        if 1 in results_flag:
            idx = results_flag.index(1)
            ret_dict["start"].append({"tid": tid, "tracklet": tracklet, "frame_id": results_fid[idx][0]})
            print("start: ", {"tid": tid, "frame_id": results_fid[idx][0]})
        elif 2 in results_flag:
            idx = results_flag.index(2)
            ret_dict["stop"].append({"tid": tid, "tracklet": tracklet, "frame_id": results_fid[idx][1]})
            print("stop: ", {"tid": tid, "frame_id": results_fid[idx][1]})
    return ret_dict


def crop_img(img, box, frame_height, frame_width):
    x1, y1, x2, y2 = list(map(int, box))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, frame_width)
    y2 = min(y2, frame_height)
    return img[y1:y2, x1:x2]


def get_appearance_feature(matching_tracklet, video_path):
    assert os.path.isfile(video_path), 'video path wrong!'
    start_tracklets = matching_tracklet["start"]
    stop_tracklets = matching_tracklet["stop"]

    cap = cv2.VideoCapture(video_path)
    frame_count = 1
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    print("image shape: ", frame_height, frame_width)
    while ret:
        for tracklets in start_tracklets:
            tracklet, tid = tracklets["tracklet"], tracklets["frame_id"]
            if frame_count in tracklet["frame_ids"]:
                idx = tracklet["frame_ids"].index(frame_count)
                img = crop_img(frame, tracklet["boxs"][idx], frame_height, frame_width)
                # feature = get_hog_features(img, winSize, blockSize, blockStride, cellSize, nbins)
                feature = get_bgr_features(img)
                tracklet["features"][idx] = feature
                if feature.shape[0] == 0:
                    print("start dont extract feature", idx, frame_count, tracklet["boxs"][idx])
        for tracklets in stop_tracklets:
            tracklet, tid = tracklets["tracklet"], tracklets["frame_id"]
            if frame_count in tracklet["frame_ids"]:
                idx = tracklet["frame_ids"].index(frame_count)
                img = crop_img(frame, tracklet["boxs"][idx], frame_height, frame_width)
                # feature = get_hog_features(img, winSize, blockSize, blockStride, cellSize, nbins)
                feature = get_bgr_features(img)
                if feature.shape[0] == 0:
                    print("start dont extract feature", idx, frame_count, tracklet["boxs"][idx])
                tracklet["features"][idx] = feature
        frame_count += 1
        ret, frame = cap.read()

def get_classified_tracklets(tracklet_dict, camera_config):
    ret_dict = {}
    cj = CrossJudger()
    for tid, tracklet in tracklet_dict.items():
        results_flag = []
        results_fid = []
        for m_name, value in camera_config.items():
            cross_flag_start = False
            cross_flag_stop = False
            """
            True True -> 0
            True False -> 1
            False True -> 2
            False False -> 3
            """
            start_fid, stop_fid = -1, -1
            for tlb, fid in zip(tracklet["boxs"], tracklet["frame_ids"]):
                start_flag = cj.judge(value["src"]["point_1"], value["src"]["point_2"], tlb)
                cross_flag_start = cross_flag_start or start_flag
                if start_flag:
                    start_fid = fid
                    continue
                stop_flag = cj.judge(value["dst"]["point_1"], value["dst"]["point_2"], tlb)
                # cj.visualize(value["dst"]["point_1"], value["dst"]["point_2"], tlb, cross_flag=stop_flag)
                cross_flag_stop = cross_flag_stop or stop_flag
                if stop_flag:
                    stop_fid = fid
            if cross_flag_start and cross_flag_stop:
                results_flag.append(0)
                results_fid.append([start_fid, stop_fid, m_name])
            elif cross_flag_start and not cross_flag_stop:
                results_flag.append(1)
                results_fid.append([start_fid, stop_fid, m_name])
            elif not cross_flag_start and cross_flag_stop:
                results_flag.append(2)
                results_fid.append([start_fid, stop_fid, m_name])
            else:
                results_flag.append(3)
                results_fid.append([start_fid, stop_fid, m_name])
        if 0 in results_flag:
            idx = results_flag.index(0)
            key = results_fid[idx][-1]
            if key in ret_dict.keys():
                ret_dict[key].append(tracklet)
            else:
                ret_dict[key] = [tracklet]
    return ret_dict