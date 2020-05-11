import os
import sys
import json
import cv2
import numpy as np
from hausdorff_dist import hausdorff_distance

def check_bbox_inside_with_roi(bbox, mask):
    #check if four point of bbox all in roi area
    is_inside = True

    x_tl = bbox[1]
    y_tl = bbox[2]
    x_br = bbox[3]
    y_br = bbox[4]

    for x in [x_tl, x_br]:
        if x <= 0 or x >= mask.shape[1]:
            return False

    for y in [y_tl, y_br]:
        if y <= 0 or y >= mask.shape[0]:
            return False

    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) < (128, 128, 128):
            is_inside = False
            return is_inside

    return is_inside

def check_tracks_with_roi(tracks, mask):
    tracks_end_in_roi = []
    tracks_start_in_roi = []
    tracks_too_short = []

    for trackid, track in tracks.items():
        start_bbox = track['bbox'][0]
        end_bbox = track['bbox'][-1]

        if check_bbox_inside_with_roi(start_bbox, mask) == True:
            if track['startframe'] > 3:
                tracks_start_in_roi.append(trackid)

        if check_bbox_inside_with_roi(end_bbox, mask) == True:
            tracks_end_in_roi.append(trackid)

        if track['endframe'] - track['startframe'] < 10:
            if trackid not in tracks_start_in_roi:
                tracks_too_short.append(trackid)
    return tracks_end_in_roi, tracks_start_in_roi, tracks_too_short


def check_bbox_overlap_with_roi(bbox, mask):
    is_overlap = False
    if bbox[1] >= mask.shape[1] or bbox[2] >= mask.shape[0] \
            or bbox[3] < 0 or bbox[4] < 0:
        return is_overlap

    x_tl = bbox[1] if bbox[1] > 0 else 0
    y_tl = bbox[2] if bbox[2] > 0 else 0
    x_br = bbox[3] if bbox[3] < mask.shape[1] else mask.shape[1] - 1
    y_br = bbox[4] if bbox[4] < mask.shape[0] else mask.shape[0] - 1
    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) > (128, 128, 128):
            is_overlap = True
            return is_overlap

    return is_overlap

def is_same_direction(traj1, traj2, angle_thr):
    vec1 = np.array([traj1[-1][0] - traj1[0][0], traj1[-1][1] - traj1[0][1]])
    vec2 = np.array([traj2[-1][0] - traj2[0][0], traj2[-1][1] - traj2[0][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return False
    cos = vec1.dot(vec2)/(L1*L2)
    angle = np.arccos(cos) * 360/(2*np.pi)
    if angle < angle_thr:
        return True
    else:
        return False

def calc_angle(vec1, vec2):
    vec1 = np.array([traj1[-1][0] - traj1[-5][0], traj1[-1][1] - traj1[-5][1]])
    vec2 = np.array([traj2[-1][0] - traj2[-5][0], traj2[-1][1] - traj2[-5][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return 90
    cos = vec1.dot(vec2)/(L1*L2)
    if cos > 1:
        return 90
    angle = np.arccos(cos) * 360/(2*np.pi)
    return angle

def count_video(data_root, video_name, save_root):
    '''
    data_root: the path which contains (1)the track_results(2)masks(3)AIC20_track1(4)cam-configs
    video_name: the video to count
    save_root: the root path to save counting results
    '''
    #get cam_name related index
    #for datasetB, you shold change the path for video_list_id.txt 
    video_list_file = os.path.join(data_root, 'AIC20_track1/Dataset_A/list_video_id.txt')
    if not os.path.exists(video_list_file):
        print('no list_video_id.txt found! ')
        return
        
    with open(video_list_file, 'r') as fl:
        lines = [line.strip('\n').split(' ') for line in fl]
        for line in lines:
            if line[1][:-4] != video_name:
                continue
            else:
                video_idx = line[0]

    # load movements tipical trajs
    cam_name = 'cam_' + video_name.split('_')[1]
    cam_conf = os.path.join(data_root, 'cam-configs', cam_name+'.json')
    tipical_trajs = {}
    with open(cam_conf, 'r') as fc:
        movements = json.load(fc)
        for movement_id, movement_info in movements.items():
            try:
                tracklets = movement_info['tracklets']
                tipical_trajs[movement_id] = tracklets
            except:
                print('cam config error!')
                return

    #read mask image
    cam_mask = os.path.join(data_root, 'mask', video_name+'.jpg')
    mask = cv2.imread(cam_mask)
    h, w, c = mask.shape
    #load tracks
    tracks = {}
    #path to track results path
    track_file = os.path.join(data_root, 'track_results', video_name+'.txt')
    with open(track_file, 'r') as ft:
        lines = [line.strip('\n').split(' ') for line in ft]
        for line in lines:
            frameid = int(line[0])
            trackid = int(line[1])
            x1 = int(float(line[2]))
            y1 = int(float(line[3]))
            x2 = int(float(line[4]))
            y2 = int(float(line[5]))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            label = line[6]
            if trackid in tracks:
                tracks[trackid]['endframe'] = frameid
                tracks[trackid]['bbox'].append([frameid, x1, y1, x2, y2, label])
                tracks[trackid]['tracklet'].append([cx, cy])
            else:
                tracks[trackid] = {'startframe' : frameid,
                                   'endframe' : frameid,
                                   'bbox' : [[frameid, x1, y1, x2, y2, label]],
                                   'tracklet' : [[cx, cy]]}

    #split tracklets
    tracks_end_in_roi, tracks_start_in_roi, tracks_too_short = check_tracks_with_roi(tracks, mask)

    trackids = sorted([k for k in tracks.keys()])
    #save count results
    os.makedirs(save_root, exist_ok=True)
    savefile = os.path.join(save_root, video_name+'.txt')
    dst_out = open(savefile, 'w')

    #start counting
    dist_thr = 300
    angle_thr = 30
    min_length = 10
    results = []
    for trackid in trackids:
        if len(tracks[trackid]['tracklet']) < min_length:
            continue
        track_traj = tracks[trackid]['tracklet']
        #calc hausdorff dist with tipical trajs, assign the movement with the min dist
        all_dists_dict = {k: float('inf') for k in tipical_trajs}
        for m_id, m_t in tipical_trajs.items():
            for t in m_t:
                tmp_dist = hausdorff_distance(np.array(track_traj), np.array(t), distance='euclidean')
                if tmp_dist < all_dists_dict[m_id]:
                    all_dists_dict[m_id] = tmp_dist

        #check direction
        all_dists = sorted(all_dists_dict.items(), key=lambda k: k[1])
        min_idx, min_dist = None, dist_thr
        for i in range(0, len(all_dists)):
            m_id = all_dists[i][0]
            m_dist = all_dists[i][1]
            if m_dist >= dist_thr: #if min dist > dist_thr, will not assign to any movement
                break
            else:
                if is_same_direction(track_traj, tipical_trajs[m_id][0], angle_thr): #check direction
                    min_idx = m_id
                    min_dist = m_dist
                    break #if match, end
                else:
                    continue #direction not matched, find next m_id

        #cam_13 14 will not use shape based method
        direct_match_videos = ['cam_13', 'cam_14']
        if cam_name not in direct_match_videos and min_idx == None and min_dist >= dist_thr:
            continue

        #spatial constrains
        #-----------------------------------------------------------------------------------
        #cam 1
        if cam_name == 'cam_1' and min_idx in ['movement_1', 'movement_2', 'movement_3'] and \
                len(track_traj) < 30:
            continue
        #is mv3
        if cam_name == 'cam_1' and min_idx != 'movement_3':
            cx, cy = track_traj[0]
            if cx > w * 0.5 and cy > h * 0.3:
                min_idx = 'movement_3'
        #is mv1 or 2 not 3
        if cam_name == 'cam_1' and min_idx == 'movement_3':
            cx, cy = track_traj[0]
            if cx < w * 0.5 or cy < h * 0.3:
                if is_same_direction(track_traj, tipical_trajs['movement_1'][0], 45):
                    min_idx = 'movement_1'
                elif is_same_direction(track_traj, tipical_trajs['movement_2'][0], 45):
                    min_idx = 'movement_2'

        #-----------------------------------------------------------------------------------
        #cam 2
        if cam_name == 'cam_2' and min_idx in ['movement_1', 'movement_2'] and \
                len(track_traj) < 30:
                    continue
        #-----------------------------------------------------------------------------------
        #cam_3
        #is mv4
        if cam_name == 'cam_3' and min_idx != 'movement_4':
            cx, cy = track_traj[0]
            if cx > w * 0.5 and cy > h * 0.5:
                min_idx = 'movement_4'
        #is mv 1 2 3 not mv4
        if cam_name == 'cam_3' and min_idx == 'movement_4':
            cx, cy = track_traj[0]
            if cx < w * 0.5 or cy < h * 0.5:
                if is_same_direction(track_traj, tipical_trajs['movement_1'][0], 45):
                    min_idx = 'movement_1'
                elif is_same_direction(track_traj, tipical_trajs['movement_2'][0], 45):
                    min_idx = 'movement_2'
                elif is_same_direction(track_traj, tipical_trajs['movement_3'][0], 45):
                    min_idx = 'movement_3'
        #-----------------------------------------------------------------------------------
        #cam4 5
        # is mv5 not mv 1 12
        if cam_name in ["cam_4", 'cam_5'] and min_idx in ["movement_1", 'movement_12']:
            cx, cy = track_traj[0]
            if cx > 300:
                if all_dists_dict['movement_5'] < dist_thr:
                    min_idx = 'movement_5'
                else:
                    continue
        #is mv8 not mv4
        if cam_name in ['cam_4', 'cam_5'] and min_idx == 'movement_4':
            cx, cy = track_traj[0]
            if cx > 1000:
                if all_dist_dict['movement_8'] < dist_thr:
                    min_idx = 'movement_8'
                else:
                    continue
        #is mv5 not mv4
        if cam_name in ['cam_4', 'cam_5'] and min_idx == 'movement_4':
            #print(trackid)
            cx, cy = track_traj[-1]
            if cx < 100:
                if all_dist_dict['movement_5'] < dist_thr:
                    min_idx = 'movement_5'
                else:
                    continue
        if cam_name in ['cam_4', 'cam_5'] and min_idx in ['movement_3'] and len(track_traj) < 30:
            continue
        if cam_name in ['cam_4', 'cam_5'] and min_idx in ['movement_5'] and \
                trackid in tracks_start_in_roi:
            continue

        #------------------------------------------------------------------------------------
        if cam_name == 'cam_7' and trackid in tracks_end_in_roi:
            continue

        #------------------------------------------------------------------------------------
        #cam 10 11
        if cam_name in ['cam_10', 'cam_11'] and min_idx == 'movement_1':
            if track_traj[0][0] > w * 0.5:
                continue
        if cam_name in ['cam_10', 'cam_11'] and min_idx == 'movement_3':
            if track_traj[-1][0] < 200:
                continue
        #------------------------------------------------------------------------------------
        #cam12
        if cam_name == 'cam_12' and min_idx in ['movement_1', 'movement_2']:
            if track_traj[0][0] > w * 0.6 or track_traj[0][0] < 100 or track_traj[0][1] > h * 0.5:
                continue
            if len(track_traj) < 30:
                continue
            #if min_idx == 'movement_2' and track_traj[-1][0] and track_traj[-1][1] < 600:
            #    print(trackid, track_traj[-1])
        #------------------------------------------------------------------------------------
        #cam13
        if cam_name == 'cam_13':
            if track_traj[0][1] > 374 or track_traj[0][0] > w * 0.7 or track_traj[0][1] > h * 0.8:
                continue
            if track_traj[-1][0] < w * 0.7 and track_traj[-1][1] > h * 0.8:
                min_idx = 'movement_2'
            elif track_traj[-1][0] > w * 0.7 and track_traj[-1][1] > h * 0.5:
                min_idx = 'movement_3'
            elif track_traj[-1][0] < w * 0.3 and track_traj[-1][1] < h * 0.8:
                min_idx = 'movement_1'
            else:
                continue
        #------------------------------------------------------------------------------------
        #cam14
        if cam_name == 'cam_14':
            if track_traj[-1][0] <= w * 0.5 and is_same_direction(track_traj,
                    tipical_trajs['movement_1'][0], 45):
                min_idx = 'movement_1'
            elif track_traj[-1][0] > w * 0.5 and is_same_direction(track_traj, tipical_trajs['movement_2'][0], 45):
                min_idx = 'movement_2'
            elif abs(track_traj[-1][0] - track_traj[0][0]) < 100:
            # for id switch in remote area which lead to the trackid go back 
                pass
            else:
                continue
        #------------------------------------------------------------------------------------
        if cam_name == 'cam_15':
            #remove still vehicle
            track_w = tracks[trackid]['bbox'][0][3] - tracks[trackid]['bbox'][0][1]
            if abs(track_traj[-1][0] - track_traj[0][0]) < 2*track_w:
                continue
            if track_traj[0][0] < w * 0.5:
                min_idx = 'movement_1'
            elif track_traj[0][0] > w * 0.7:
                min_idx = 'movement_2'
            else:
                continue
        #------------------------------------------------------------------------------------
        #cam_20
        if cam_name in ['cam_19', 'cam_20']:
            #remove still vehicle
            track_h = tracks[trackid]['bbox'][0][4] - tracks[trackid]['bbox'][0][2]
            if abs(track_traj[-1][1] - track_traj[0][1]) < track_h:
                continue

        #save counting results
        mv_idx = min_idx.split('_')[1]
        #get last frameid in roi
        bboxes = tracks[trackid]['bbox']
        bboxes.sort(key=lambda x: x[0])

        dst_frame = bboxes[0][0]
        last_bbox = bboxes[-1]
        if check_bbox_overlap_with_roi(last_bbox, mask) == True:
            dst_frame = last_bbox[0]
        else:
            for i in range(len(bboxes) - 2, 0, -1):
                bbox = bboxes[i]
                if check_bbox_overlap_with_roi(bbox, mask) == True:
                    dst_frame = bbox[0]
                    break
                else:
                    continue

        track_types = [k[5] for k in bboxes]
        track_type = max(track_types, key=track_types.count)

        if track_type == 'car':
            cls_id = 1
        else:
            cls_id = 2
        results.append([video_idx, dst_frame, mv_idx, cls_id])
    #save
    results.sort(key=lambda x: (x[1], x[2]))
    for res in results:
        res_str = " ".join(str(k) for k in res)
        print(res_str, file=dst_out)
    dst_out.close()
    print('vehicle counting done.')


if __name__ == '__main__':
    data_root = '../'
    save_root = './vehicle_counting_results'
    #video_list = os.listdir(os.path.join(data_root, 'imageset'))
    video_list = [sys.argv[1]]

    for video_name in video_list:
        print('start to counting video %s ... ' % video_name)
        count_video(data_root, video_name, save_root)
