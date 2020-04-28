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
pid_map = {}
pid_cam_count = {}
write_lines = []

all_ids = set()

for s in itemlist:
    # ---------first we read the camera and ID info and rename the images.
    name = s.attributes['imageName'].value


    vid = s.attributes['vehicleID'].value
    vid = int(vid)

    remap_id_fill = str(vid).zfill(6)
    cam = s.attributes['cameraID'].value
    if vid not in pid_cam_count.keys():
        pid_cam_count[vid] = dict()
    if cam not in pid_cam_count[vid].keys():
        pid_cam_count[vid][cam] = 0
    all_ids.append(vid)
    save_name = '{}_{}_{}.jpg'.format(remap_id_fill, cam, pid_cam_count[vid][cam])
    

    src_path = train_path + name
    dst_path = save_path + save_name #train_all
    copyfile(src_path, dst_path)

    pid_cam_count[vid][cam] += 1
    write_lines.append(save_name)
# fid2.close()

def sort_helper(line):
    name = line.split(' ')[0]
    vid = int(name.split('_')[0])
    cam = int(name.split('_')[1][1:])
    count = int(name.split('_')[2][:-4])
    return vid, cam, count

    
fid = open('real_trainval_list.txt','w')
write_lines_sort = sorted(write_lines, key=sort_helper    )
for each in write_lines_sort:
    fid.write(each+'\n')
fid.close()



