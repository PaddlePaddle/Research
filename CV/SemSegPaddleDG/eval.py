# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99"
import logging
import math
import numpy as np
import shutil
import argparse
import pprint
from PIL import ImageOps, Image, ImageEnhance, ImageFilter
from datetime import datetime

import paddle
import paddle.fluid as fluid
from src.models.modeling.pspnet import PSPNet
from src.models.modeling.glore import Glore
from src.datasets.cityscapes import cityscapes_eval
from src.datasets.pascal_context import pascal_context_eval
from src.utils.solver import Lr
from src.utils.palette import get_palette
from src.utils.config import cfg
from src.utils.iou import IOUMetric

#  globals
data_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
data_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def print_info(*msg):
    if cfg.TRAINER_ID == 0:
        print(*msg)

def parse_args():
    parser = argparse.ArgumentParser(description='semseg-paddle')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess I/O or not',
        action='store_true',
        default=False)
    parser.add_argument(
        '--log_steps',
        dest='log_steps',
        help='Display logging information at every log_steps',
        default=10,
        type=int)
    parser.add_argument(
        '--debug',
        dest='debug',
        help='debug mode, display detail information of training',
        action='store_true')
    parser.add_argument(
        '--use_tb',
        dest='use_tb',
        help='whether to record the data during training to Tensorboard',
        action='store_true')
    parser.add_argument(
        '--tb_log_dir',
        dest='tb_log_dir',
        help='Tensorboard logging directory',
        default=None,
        type=str)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Evaluation models result on every new checkpoint',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    return parser.parse_args()



def pad_single_image(image, crop_size):
    w, h = image.size
    pad_h = crop_size - h if h < crop_size else 0
    pad_w = crop_size - w if w < crop_size else 0
    image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    assert (image.size[0] >= crop_size and image.size[1] >= crop_size)
    return image


def crop_image(image, h0, w0, h1, w1):
    return image.crop((w0, h0, w1, h1))


def flip_left_right_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def resize_image(image, out_h, out_w, mode=Image.BILINEAR):
    return image.resize((out_w, out_h), mode)


def mapper_image(image):
    image_array = np.array(image)
    image_array = image_array.transpose((2, 0, 1))
    image_array = image_array / 255.0
    image_array = (image_array - data_mean) / data_std
    image_array = image_array.astype('float32')
    image_array = image_array[np.newaxis, :]
    return image_array


def get_model(cfg):
    if cfg.MODEL.MODEL_NAME.lower() == 'pspnet':
        model = PSPNet(cfg.TRAIN.PRETRAINED_MODEL_FILE, cfg.DATASET.DATA_DIM, cfg.DATASET.NUM_CLASSES, multi_grid=cfg.MODEL.BACKBONE_MULTI_GRID)
    elif cfg.MODEL.MODEL_NAME.lower() == 'glore':
        model = Glore(cfg.TRAIN.PRETRAINED_MODEL_FILE, cfg.DATASET.DATA_DIM, cfg.DATASET.NUM_CLASSES, multi_grid=cfg.MODEL.BACKBONE_MULTI_GRID)
    return model


def get_data(cfg, split, base_size, crop_size, batch_size=1, gpu_num=1):
    if cfg.DATASET.DATASET_NAME.lower() in ['cityscapes', 'pascal_context']:
        dataset_name = cfg.DATASET.DATASET_NAME.lower()
        data_loader = globals()[dataset_name + '_' + split]
        return data_loader(root=cfg.DATASET.DATA_ROOT,
                            base_size=base_size,
                            crop_size=crop_size,
                            scale=True,
                            xmap=True,
                            batch_size=batch_size,
                            gpu_num=gpu_num)
    else:
        raise ValueError('Dataset is not supported, please check!')


def copy_model(path, new_path):
    shutil.rmtree(new_path, ignore_errors=True)
    shutil.copytree(path, new_path)
    model_path = os.path.join(new_path, '__model__')
    if os.path.exists(model_path):
        os.remove(model_path)


def mean_iou(pred, label, num_classes=19):
    label = fluid.layers.elementwise_min(fluid.layers.cast(label, np.int32),
                                         fluid.layers.assign(np.array([num_classes], dtype=np.int32)))
    label_ig = (label == num_classes).astype('int32')
    label_ng = (label != num_classes).astype('int32')
    pred = fluid.layers.cast(fluid.layers.argmax(pred, axis=1), 'int32')
    pred = pred * label_ng + label_ig * num_classes
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
    label.stop_gradient = True
    return miou, wrong, correct


def eval(cfg):
    with fluid.dygraph.guard():
        num_classes = cfg.DATASET.NUM_CLASSES
        base_size = cfg.TEST.BASE_SIZE
        crop_size = cfg.TEST.CROP_SIZE
        multi_scales = cfg.TEST.MULTI_SCALE
        flip = cfg.TEST.FLIP

        if not multi_scales:
            scales = [1.0]
        else:
            # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]
            scales = [0.5, 0.75, 1.0, 1.25, 1.35, 1.5, 1.75, 2.0, 2.2]  # maybe work better

        if len(scales) == 1:  # single scale
            # stride_rate = 2.0 / 3.0
            stride_rate = 1.0 / 2.0  # maybe work better
        else:
            stride_rate = 1.0 / 2.0
        stride = int(crop_size * stride_rate)  # slide stride

        model = get_model(cfg)
        x = np.random.randn(1, 3, 224, 224).astype('float32')
        x = fluid.dygraph.to_variable(x)
        y = model(x)
        iou = IOUMetric(num_classes)
        model_path = cfg.TEST.TEST_MODEL_FILE
        
        assert os.path.exists(model_path), "your_model '{}' is not exists".format(
               model_path)
        print("successfully load model:  {} !".format(model_path))
        model_param, _ = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(model_param)

        assert len(model_param) == len(model.state_dict()), "The number of parameters is not equal. Loading parameters failed, \
            Please check whether the model is consistent!"
        model.eval()

        prev_time = datetime.now()
        reader = get_data(cfg=cfg,
                      split='eval', 
                      base_size=cfg.TEST.BASE_SIZE, 
                      crop_size=cfg.TEST.CROP_SIZE)

        print('MultiEvalModule: base_size= {}, crop_size= {}, scales= {}'.format(base_size, crop_size, scales))
        print('conducting validation ...')
        logging.basicConfig(level=logging.INFO,
                            filename='{}_{}_eval_dygraph.log'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info(cfg.MODEL.MODEL_NAME)
        logging.info(cfg)
        palette = get_palette(cfg.DATASET.DATASET_NAME)
        count = 0
        total = cfg.DATASET.VAL_TOTAL_IMAGES
        for data in reader():
            image = data[0]
            label_path = data[2]  # val_label is a picture, test_label is a path
            print("label_path: ", label_path)
            if cfg.DATASET.DATASET_NAME == "cityscapes":
                postfix = label_path.split('/')[-2:]
                save_png_dir = str(cfg.TEST.OUTPUT_PATH) + '/' + str(cfg.MODEL.MODEL_NAME) +'_pred_' + str(cfg.DATASET.DATASET_NAME) +'/' + postfix[0] 
                if not os.path.exists(save_png_dir):
                    os.makedirs(save_png_dir)
                save_png_path = save_png_dir + '/' + postfix[1]
            elif cfg.DATASET.DATASET_NAME == "pascal_context":
                postfix = label_path.split('/')[-1]
                save_png_dir = str(cfg.TEST.OUTPUT_PATH) + '/' + str(cfg.MODEL.MODEL_NAME) +'_pred_' + str(cfg.DATASET.DATASET_NAME)
                if not os.path.exists(save_png_dir):
                    os.makedirs(save_png_dir)
                save_png_path = save_png_dir + '/' + postfix

            else:
                save_png_path = None
            print("save_png_path: ", save_png_path)
            label_np = data[1]
            w, h = image.size  # h 1024, w 2048
            scores = np.zeros(shape=[num_classes, h, w], dtype='float32')
            for scale in scales:
                long_size = int(math.ceil(base_size * scale))  # long_size
                if h > w:
                    height = long_size
                    width = int(1.0 * w * long_size / h + 0.5)
                    short_size = width
                else:
                    width = long_size
                    height = int(1.0 * h * long_size / w + 0.5)
                    short_size = height

                cur_img = resize_image(image, height, width)
                # pad
                if long_size <= crop_size:
                    pad_img = pad_single_image(cur_img, crop_size)
                    # pad_img = cur_img
                    pad_img = mapper_image(pad_img)
                    pad_img = fluid.dygraph.to_variable(pad_img)
                    pred1, pred2 = model(pad_img)
                    pred1 = pred1.numpy()
                    outputs = pred1[:, :, :height, :width]
                    if flip:
                        pad_img_filp = flip_left_right_image(cur_img)
                        pad_img_filp = pad_single_image(pad_img_filp, crop_size)  # pad
                        pad_img_filp = mapper_image(pad_img_filp)
                        pad_img_filp = fluid.dygraph.to_variable(pad_img_filp)
                        pred1, pred2 = model(pad_img_filp)
                        pred1 = fluid.layers.reverse(pred1, axis=3)
                        pred1 = pred1.numpy()
                        outputs += pred1[:, :, :height, :width]
                else:
                    if short_size < crop_size:
                        # pad if needed
                        pad_img = pad_single_image(cur_img, crop_size)
                    else:
                        pad_img = cur_img
                    pw, ph = pad_img.size
                    assert (ph >= height and pw >= width)

                    # slide window
                    h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                    w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                    outputs = np.zeros(shape=[1, num_classes, ph, pw], dtype='float32')
                    count_norm = np.zeros(shape=[1, 1, ph, pw], dtype='int32')
                    for idh in range(h_grids):
                        for idw in range(w_grids):
                            h0 = idh * stride
                            w0 = idw * stride
                            h1 = min(h0 + crop_size, ph)
                            w1 = min(w0 + crop_size, pw)
                            crop_img = crop_image(pad_img, h0, w0, h1, w1)
                            pad_crop_img = pad_single_image(crop_img, crop_size)
                            pad_crop_img = mapper_image(pad_crop_img)
                            pad_crop_img = fluid.dygraph.to_variable(pad_crop_img)
                            pred1, pred2 = model(pad_crop_img)  # shape [1, num_class, h, w]
                            pred = pred1.numpy()  # channel, h, w
                            outputs[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                            count_norm[:, :, h0:h1, w0:w1] += 1
                            if flip:
                                pad_img_filp = flip_left_right_image(crop_img)
                                pad_img_filp = pad_single_image(pad_img_filp, crop_size)  # pad
                                pad_img_array = mapper_image(pad_img_filp)
                                pad_img_array = fluid.dygraph.to_variable(pad_img_array)
                                pred1, pred2 = model(pad_img_array)
                                pred1 = fluid.layers.reverse(pred1, axis=3)
                                pred = pred1.numpy()
                                outputs[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                                count_norm[:, :, h0:h1, w0:w1] += 1
                    assert ((count_norm == 0).sum() == 0)
                    outputs = outputs / count_norm
                    outputs = outputs[:, :, :height, :width]
                outputs = fluid.dygraph.to_variable(outputs)
                outputs = fluid.layers.resize_bilinear(outputs, out_shape=[h, w])
                score = outputs.numpy()[0]
                scores += score  # the sum of all scales, shape: [channel, h, w]
                pred = np.argmax(score, axis=0).astype('uint8')
                picture_path = '{}'.format(save_png_path).replace('.png', '_scale_{}'.format(scale))
                print("picture_path: ", picture_path)
                save_png(pred, palette, picture_path)
            pred = np.argmax(scores, axis=0).astype('uint8')
            picture_path = '{}'.format(save_png_path).replace('.png', '_scores')
            save_png(pred, palette, picture_path)
            print("pred.min={}, max= {}: ", np.min(pred), np.max(pred))
            print("label_np.min={}, max= {} ", np.min(label_np), np.max(label_np)) 
            iou.add_batch(pred, label_np)  # cal iou
            count += 1
            print('Processing..........[{}/{}]'.format(count, total))
        print('eval done!')
        logging.info('eval done!')
        acc, acc_cls, iu, mean_iu, fwavacc, kappa = iou.evaluate()
        logging.info('acc = {}'.format(acc))
        logging.info('acc_cls = {}'.format(acc_cls))
        logging.info('iu = {}'.format(iu))
        logging.info('mean_iou --255 = {}'.format(mean_iu))
        print('mean_iou = {}'.format(np.nanmean(iu[:-1])))  # realy iou
        logging.info('mean_iou = {}'.format(np.nanmean(iu[:-1])))
        logging.info('fwavacc = {}'.format(fwavacc))
        logging.info('kappa = {}'.format(kappa))
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        logging.info('time of validation ' + time_str)


def save_png(pred_value, palette, name):
    if isinstance(pred_value, np.ndarray):
        if pred_value.ndim == 3:
            batch_size = pred_value.shape[0]
            if batch_size == 1:
                pred_value = pred_value.squeeze(axis=0)
                image = Image.fromarray(pred_value).convert('P')
                image.putpalette(palette)
                save_path = '{}.png'.format(name)
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                image.save(save_path)
            else:
                for batch_id in range(batch_size):
                    value = pred_value[batch_id]
                    image = Image.fromarray(value).convert('P')
                    image.putpalette(palette)
                    save_path = '{}.png'.format(name[batch_id])
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    image.save(save_path)
        elif pred_value.ndim == 2:
            image = Image.fromarray(pred_value).convert('P')
            image.putpalette(palette)
            save_path = '{}.png'.format(name)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image.save(save_path)
    else:
        raise ValueError('Only support nd-array')


if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print_info(pprint.pformat(cfg))
    eval(cfg)

