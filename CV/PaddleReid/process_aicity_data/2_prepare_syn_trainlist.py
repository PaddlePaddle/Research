import os
from shutil import copyfile
from xml.dom import minidom
import numpy as np
import pdb
import pickle


train_path = 'image_train/'
save_path = './aicity20_all/image_train/'

#---------prepare training all set-------
xmldoc = minidom.parse('train_label_utf8.xml')
itemlist = xmldoc.getElementsByTagName('Items')[0]
itemlist = itemlist.getElementsByTagName('Item')

count = 0
id_start = 666 ### 1-666 is real data vid
pid_map = {}
pid_cam_count = {}
write_lines = []

max_angle = 0
min_angle = 0
all_cams = set()
for s in itemlist:
    # ---------first we read the camera and ID info and rename the images.
    name = s.attributes['imageName'].value
    color = s.attributes['colorID'].value
    cartype = s.attributes['typeID'].value
    orientation = s.attributes['orientation'].value
    orientation = int(float(orientation))
    max_angle = max(orientation, max_angle)
    min_angle = min(orientation, min_angle)
    
    
    vid = s.attributes['vehicleID'].value
    vid = int(vid)
    vid = vid + id_start

    remap_id_fill = str(vid).zfill(6)
    cam = s.attributes['cameraID'].value
    all_cams.add(cam)
    if vid not in pid_cam_count.keys():
        pid_cam_count[vid] = dict()
    if cam not in pid_cam_count[vid].keys():
        pid_cam_count[vid][cam] = 0
    
    save_name = '{}_{}_{}_{}_{}_{}.jpg'.format(remap_id_fill, cam, color, cartype, orientation, pid_cam_count[vid][cam])
    print(save_name)
    src_path = train_path + name
    dst_path = save_path + save_name
    copyfile(src_path, dst_path)

    pid_cam_count[vid][cam] += 1

    write_lines.append(save_name)

print(max_angle)
print(min_angle)
print(all_cams)
pdb.set_trace()
fid = open('syn_trainval_list.txt','w')
def sort_helper(line):
    name = line.split(' ')[0]
    vid = int(name.split('_')[0])
    cam = int(name.split('_')[1][1:])
    color = int(name.split('_')[2])
    cartype = int(name.split('_')[3])
    orientation = int(name.split('_')[4])
    count = int(name.split('_')[5][:-4])
    return vid, cam, count, color, cartype, orientation

write_lines_sort = sorted(write_lines, key=sort_helper    )
for each in write_lines_sort:
    fid.write(each+'\n')
fid.close()

