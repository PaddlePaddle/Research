import os
import sys
import cv2
import numpy as np

def get_roi_points_list(roi_file):
    roi_points_list = []
    with open(roi_file, 'r') as f:
        full_lines = [line.strip('\n').split(',') for line in f]
        for line in full_lines:
            x = int(line[0])
            y = int(line[1])
            roi_points_list.append([x, y])
    #print(roi_points_list)
    return np.array(roi_points_list)

def generate_mask_for_video(data_root):
    mask_path = os.path.join(data_root, 'mask')
    os.makedirs(mask_path, exist_ok=True)

    video_id_list = os.path.join(data_root, 'AIC20_track1/Dataset_A/list_video_id.txt')
    print(video_id_list)
    video_list = [k.strip('\n').split(' ')[-1][:-4] for k in open(video_id_list).readlines()]
    print(video_list)

    for video_name in video_list:
        cam_name = '_'.join(k for k in video_name.split('_')[:2])
        video_roi_file = os.path.join(data_root, 'AIC20_track1/ROIs', cam_name + '.txt')
        video_roi_area = get_roi_points_list(video_roi_file)

        video_frame0 = os.path.join(data_root, 'imageset', video_name, '00001.jpg')
        f0_img = cv2.imread(video_frame0)
        h, w, c = f0_img.shape
        mask_mat = np.zeros((h, w, c), np.uint8)
        cv2.fillPoly(mask_mat, [video_roi_area], (255, 255, 255))
        mask_jpg = os.path.join(mask_path, video_name+'.jpg')
        cv2.imwrite(mask_jpg, mask_mat)

    print('done.')

def is_outside_roi(bbox, mask):
    is_outside = True

    x_tl = bbox[0] if bbox[0] > 0 else 0
    y_tl = bbox[1] if bbox[1] > 0 else 0
    x_br = bbox[2] if bbox[2] < mask.shape[1] else mask.shape[1] - 1
    y_br = bbox[3] if bbox[3] < mask.shape[0] else mask.shape[0] - 1
    #print(x_tl, y_tl, x_br, y_br)
    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) > (128, 128, 128):
            is_outside = False
            return is_outside
    return is_outside

def check_bboxes(data_root, save_root):
    os.makedirs(save_root, exist_ok=True)
    det_files = os.listdir(data_root)
    #det_files = ['cam_1.txt']
    for det_file in det_files:
        video_name = det_file[:-4]
        print(video_name)
        mask_file = './mask/' + video_name + '.jpg'
        mask = cv2.imread(mask_file)
        save_file = os.path.join(save_root, det_file)
        save_out = open(save_file, 'w')

        with open(os.path.join(data_root, det_file), 'r') as f:
            full_lines = [line.strip('\n').split(' ') for line in f]
            for line in full_lines:
                xmin = int(float(line[2]))
                ymin = int(float(line[3]))
                xmax = int(float(line[4]))
                ymax = int(float(line[5]))
                bbox = [xmin, ymin, xmax, ymax]
                #print(bbox)
                if is_outside_roi(bbox, mask) == True:
                    continue
                else:
                    line_str = " ".join(k for k in line)
                    print(line_str, file=save_out)
        save_out.close()


if __name__ == "__main__":
    #generate roi mask for tracking
    #data_root = '.'
    #generate_mask_for_video(data_root)

    #remove det results outside mask
    #det_results_path = './PaddleDetection/output/det_results_omni_res50'
    det_results_path = './PaddleDetection/output/det_results_res50'
    save_root = './det_results_check'
    check_bboxes(det_results_path, save_root)
