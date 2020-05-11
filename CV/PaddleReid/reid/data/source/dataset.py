import os
import pdb

import numpy as np

def read_txt(path):
    with open(path, 'r') as fid:
        content = [ line.strip() for line in fid.readlines()]
    return content

def get_remap_dict(content):
    remap_dict = {}
    all_pids = set()
    for each in content:
        cur_pid = int(each.split('/')[1].split('_')[0])
        all_pids.add(cur_pid)
    all_pids = list(all_pids)
    all_pids = sorted(all_pids)
    remap_dict = {all_pids[i]:i for i in range(len(all_pids))}
    #pdb.set_trace()
    return remap_dict


class Dataset(object):

    def __init__(self, root):
        self.root = root
        self.trainval = []
        self.train = []
        self.val_query, self.val_gallery = [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    def load_trainval(self, list_name='all_trainval_pids.txt', verbose=True, relabel=True):
        train_list = os.path.join(self.root, list_name)
        trainsets = read_txt(train_list)
        remap_dict = get_remap_dict(trainsets)
        train_pids = set()
        for each in trainsets:
            line_split = each.split('/')[1]
            line_split = line_split.split('_')
            
            fname = each
            pid = int(line_split[0])
            if relabel:
                pid = remap_dict[pid]
            camid = int(line_split[1][1:])
            if len(line_split)>3:
                color = int(line_split[2])
                cartype = int(line_split[3])
                direction = float(line_split[4])
            else:
                color = -1
                cartype = -1
                direction = -1
            self.train.append([fname, pid, camid, color, cartype, direction])
            train_pids.add(pid)
        self.print_statistic(self.train, train_pids, 'train' )

    def load_query(self, list_name='query_list.txt', verbose=True):
        query_list = os.path.join(self.root, list_name)
        querysets = read_txt(query_list)
        query_pids = set()
        for each in querysets:
            fname = each
            pid = int(-1)
            camid = int(-1)
            color = -1
            cartype = -1
            direction = -1
            self.query.append([fname, pid, camid, color, cartype, direction])
            query_pids.add(pid)
        self.print_statistic(self.query, query_pids, 'query' )
    
    def load_gallery(self, list_name='gallery_list.txt', verbose=True):
        gallery_list = os.path.join(self.root, list_name)
        gallerysets = read_txt(gallery_list)
        gallery_pids = set()
        for each in gallerysets:
            fname = each
            pid = int(-1)
            camid = int(-1)
            color = -1
            cartype = -1
            direction = -1
            self.gallery.append([fname, pid, camid, color, cartype, direction])
            gallery_pids.add(pid)
        self.print_statistic(self.gallery, gallery_pids, 'gallery' )

    
    def print_statistic(self, imageset, pids, set_name):
        print("  {}    | {:5d} | {:8d}"
                .format(set_name, len(pids), len(imageset)))
