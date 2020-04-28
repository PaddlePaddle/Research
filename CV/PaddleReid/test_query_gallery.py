from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os
import time
import argparse
import functools
import numpy as np
import pickle

import paddle.fluid as fluid

#from reid.dataset.dataset import Dataset
from reid.data.source.dataset import Dataset
from reid.data.reader_mt import create_readerMT
from reid.model import model_creator

from config import cfg, parse_args, print_arguments, print_arguments_dict 

def infer_feas(source, source_name, source_list, source_num, img_dir, cfg, exe, test_prog, data_loader, pred_list, places):
    reader_config = {'dataset':source, 
                     'img_dir':img_dir,
                     'batch_size':cfg.test_batch_size,
                     'num_instances':cfg.num_instances,
                     'is_test':True,
                     'sample_type':'Base',
                     'shuffle':False,
                     'drop_last':False,
                     'worker_num':8,
                     'bufsize':32,
                     'input_fields':['image','index'],
                     'cfg':cfg}
    new_reader, _, _, _ = create_readerMT(reader_config)
    data_loader.set_sample_list_generator(new_reader, places = places)


    start_time = time.time()
    fea_dict = {}
    count = 0

    for data in data_loader():
        out = exe.run(test_prog, fetch_list=[v.name for v in pred_list], feed = data, return_numpy=True)

        feas = out[0]
        cur_index = out[1].flatten().tolist()
        count = count+ len(cur_index)
        for single_index, fea in zip(cur_index, feas):
            fname = source_list[single_index]
            fea_dict[fname] = fea

        cur_time = time.time() - start_time
        start_time = time.time()
        output_str = '{}/{}imgs, time:{} '.format(count, source_num, cur_time )
        print(output_str)
    if count == source_num:
        print(source_name+' features extract Done!!!')

    return fea_dict

def merge_flip_feas(ori_feas, flip_feas):
    final_feas = {}
    for each in ori_feas.keys():
        ori_fea = ori_feas[each]
        flip_fea = flip_feas[each]
        final_fea = ori_fea + flip_fea
        final_fea_norm = np.linalg.norm(final_fea)
        final_feas[each] = final_fea / final_fea_norm
    return final_feas

def get_save_name(set_name, cfg):
    save_name = 'real_'+ set_name +'_fea_' + cfg.model_arch
    if cfg.use_crop:
        save_name += '_crop'
    if cfg.flip_test:
        save_name += '_add_flip'
    save_name +='.pkl'
    return save_name

def main(cfg):
    ReidDataset = Dataset(root = cfg.data_dir)
    ReidDataset.load_query()
    ReidDataset.load_gallery()
    query_source = ReidDataset.query
    gallery_source = ReidDataset.gallery
    query_num = len(query_source)
    gallery_num = len(gallery_source)
    query_names = [ query_source[i][0] for i in range(query_num)]
    gallery_names = [ gallery_source[i][0] for i in range(gallery_num)]

    if cfg.use_crop:
        query_img_dir = './dataset/aicity20_all/image_query_crop'
        gallery_img_dir = './dataset/aicity20_all/image_test_crop'
    else:
        query_img_dir = './dataset/aicity20_all/image_query'
        gallery_img_dir = './dataset/aicity20_all/image_gallery'


    image = fluid.layers.data(name='image', shape=[None, 3, cfg.target_height, cfg.target_width], dtype='float32')
    index = fluid.layers.data(name='index', shape=[None, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(feed_list=[image, index], capacity=128, use_double_buffer=True, iterable=True)

    model = model_creator(cfg)
    if cfg.use_multi_branch:
        fea_out = model.net_multi_branch(input=image, is_train=False, class_dim=1695, num_features = cfg.num_features)
    else:
        fea_out = model.net(input=image, is_train=False, class_dim=1695, num_features = cfg.num_features)
    index = fluid.layers.cast(index, dtype='int32') 
    fea_out = fluid.layers.l2_normalize(x=fea_out, axis=1)

    pred_list = [fea_out, index]
    test_prog = fluid.default_main_program().clone(for_test=True)
    image.persistable = True
    fea_out.persistable = True
 
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())


    load_dir = os.path.join(cfg.model_save_dir, cfg.model_arch, cfg.weights)
    print(load_dir)
    def if_exist(var):
        if os.path.exists(os.path.join(load_dir, var.name)):
            print(var.name)
            return True
        else:
            return False
    fluid.io.load_vars(
        exe, load_dir, main_program=test_prog, predicate=if_exist)


    test_prog = fluid.CompiledProgram(test_prog).with_data_parallel()


    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    print("Found {} CUDA devices.".format(devices_num))
    if devices_num==1:
        places = fluid.cuda_places(0)
    else:
        places = fluid.cuda_places()


    query_save_name = get_save_name('query', cfg)
    pdb.set_trace()
    cfg.use_flip = False
    query_feas = infer_feas(query_source, 'query', query_names, query_num, query_img_dir, cfg, exe, test_prog, data_loader, pred_list, places) 
    if cfg.flip_test:
        cfg.use_flip = True
        query_feas_flip = infer_feas(query_source, 'query', query_names, query_num, query_img_dir, cfg, exe, test_prog, data_loader, pred_list, places) 
        final_query_feas = merge_flip_feas(query_feas, query_feas_flip)
    else:
        final_query_feas = query_feas
    with open(query_save_name,'w') as fid:
        pickle.dump(final_query_feas,fid)




    gallery_save_name = get_save_name('gallery', cfg)

    cfg.use_flip = False
    gallery_feas = infer_feas(gallery_source, 'gallery', gallery_names, gallery_num, gallery_img_dir, cfg, exe, test_prog, data_loader, pred_list, places) 
    if cfg.flip_test:
        cfg.use_flip = True
        gallery_feas_flip = infer_feas(gallery_source, 'gallery', gallery_names, gallery_num, gallery_img_dir, cfg, exe, test_prog, data_loader, pred_list, places) 
        final_gallery_feas = merge_flip_feas(gallery_feas, gallery_feas_flip)
    else:
        final_gallery_feas = gallery_feas
    with open(gallery_save_name,'w') as fid:
        pickle.dump(final_gallery_feas,fid)


if __name__ == '__main__':
    args = parse_args()
    print_arguments_dict(args)
    main(args)
