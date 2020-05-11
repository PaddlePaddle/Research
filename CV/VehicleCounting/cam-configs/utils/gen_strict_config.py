# -*- coding:utf-8 -*-
import yaml
import os
import json
import random
from utils import *
gen_config_flag = True 
gen_tracklet_flag = False 
if gen_config_flag:
    # exclusive_relationship
    with open("strict_config.yaml") as f:
        yaml_str = f.read()
    yaml_config = yaml.load(yaml_str)

    config_folder_path = "../../cam-config-tj-new/"
    config_out_folder = "../../cam-config-tj-new/"
    if not os.path.exists(config_out_folder):
        os.makedirs(config_out_folder)
    for i in range(1, 21, 1):
        json_path = os.path.join(config_folder_path, "cam_{}.json".format(i))
        key = "cam_{}".format(i)
        with open(json_path) as f:
            json_config = json.load(f)
        if key not in yaml_config.keys():
            print(key, "not in yaml config.")
            continue
        for mv, value in json_config.items():
            if mv not in yaml_config[key].keys():
                print(key, mv, "not in")
                continue
            if 'src' not in yaml_config[key][mv].keys():
                continue
            value['src']["exclusive_relationship"] = yaml_config[key][mv]["src"]
            if 'dst' not in yaml_config[key][mv].keys():
                continue
            value['dst']["exclusive_relationship"] = yaml_config[key][mv]["dst"]

        with open(os.path.join(config_out_folder, "cam_{}.json".format(i)), 'w') as f:
            json.dump(json_config, f)
"""
if gen_tracklet_flag:
    # generate tracklet
    track_file_path = "../../../online/mot_txt/"
    camera_config_path = "../../cam-config-tj-new/"
    config_out_folder = "../../cam-config-tj-new/"
    file_list = os.listdir(track_file_path)
    #file_list = ["cam_9.txt", "cam_10.txt", "cam_16.txt", "cam_18.txt", "cam_19.txt"]
    file_list = ["cam_10.txt"]
    random.seed(20)
    for file_name in file_list:
        if len(file_name.split("_")) > 2:
            continue
        config_name = "_".join(file_name.split('.')[0].split('_')[:2]) + '.json'
        print("++++++++[START PROCESS {}]+++++++++".format(file_name))
        track_result_path = os.path.join(track_file_path, file_name)
        config_path = os.path.join(camera_config_path, config_name)
        print("read_files", track_result_path)
        box_dict = read_files(track_result_path)
        print("read_camera_config", config_path)
        camera_config = read_camera_config(config_path)
        print("class_tracklet", file_name)
        matching_tracklet = get_classified_tracklets(box_dict, camera_config)
        print(matching_tracklet.keys())
        for m_name, value in matching_tracklet.items():
            idx = random.randint(0, len(value)-1)
            idxs = []
            if m_name == "movement_1":
                idx = 2 
                idxs.append(idx)
            if m_name == "movement_2":
               # import pdb;pdb.set_trace()
                idx = 0 
                idxs.append(idx)
                idx = 1 
                idxs.append(idx)
            # boxs = [(int((x[0] + x[2])/2), int(x[3])) for x in value[idx]["boxs"]]
            camera_config[m_name]["tracklets"] = []
            for idx in idxs:
                boxs = [(int((x[0] + x[2])/2), int(x[3] + x[1])/2) for x in value[idx]["boxs"]]
                camera_config[m_name]["tracklets"].append(boxs)
                print("%s %s %d %d"%(file_name, m_name, len(boxs), idx))
        #if file_name == "cam_10.txt":
        #    boxs = [(int(x[0] + 100 + x[1]/3), int(x[1])) for x in camera_config["movement_2"]["tracklets"]]
        #    camera_config["movement_3"]["tracklets"] = boxs 
        with open(os.path.join(config_out_folder, config_name), 'w') as f:
            json.dump(camera_config, f)
            """


