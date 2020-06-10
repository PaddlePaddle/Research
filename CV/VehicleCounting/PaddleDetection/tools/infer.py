# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
from PIL import Image


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from PaddleDetection.ppdet.core.workspace import load_config, merge_config, create

from PaddleDetection.ppdet.utils.eval_utils import parse_fetches
from PaddleDetection.ppdet.utils.cli import ArgsParser
from PaddleDetection.ppdet.utils.check import check_gpu, check_version
from PaddleDetection.ppdet.utils.visualizer import visualize_results
import PaddleDetection.ppdet.utils.checkpoint as checkpoint

from PaddleDetection.ppdet.data.reader import create_reader

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    images = sorted(images, key=lambda x: int(x.split('/')[-1][:-4]))
    return images

def export_results_to_txt(image_path, bbox_results, save_root):
    frame_id = image_path.split('/')[-1][:-4]
    cam_id = image_path.split('/')[-2]
    #print('camid & frameid: ', cam_id, frame_id)
    save_path = os.path.join(save_root, 'det_results/')
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, cam_id + '.txt')

    with open(save_file, 'a+') as f:
        for res in bbox_results:
            if res['category_id'] == 1:
                label = 'car'
            elif res['category_id'] == 2:
                label = 'truck'
            else:
                continue

            #label = res['category_id']
            bbox = res['bbox']
            score = res['score']
            if score < 0.5:
                continue
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            info = [frame_id, -1, xmin, ymin, xmax, ymax, label, score]
            info_str = " ".join(str(k) for k in info)
            info_str += '\n'
            #print(info_str)
            f.write(info_str)

def main():
    cfg = load_config(FLAGS.config)

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    dataset = cfg.TestReader['dataset']

    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(cfg.TestReader)
    loader.set_sample_list_generator(reader, place)

    exe.run(startup_prog)
    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)

    # parse infer fetches
    assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg['metric'] in ['COCO', 'OID']:
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
        extra_keys = ['im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    # parse dataset category
    if cfg.metric == 'COCO':
        from PaddleDetection.ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg.metric == 'OID':
        from PaddleDetection.ppdet.utils.oid_eval import bbox2out, get_category_info
    if cfg.metric == "VOC":
        from PaddleDetection.ppdet.utils.voc_eval import bbox2out, get_category_info
    if cfg.metric == "WIDERFACE":
        from PaddleDetection.ppdet.utils.widerface_eval_utils import bbox2out, get_category_info

    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # use VisualDL to log image
    if FLAGS.use_vdl:
        from visualdl import LogWriter
        vdl_writer = LogWriter(FLAGS.vdl_log_dir)
        vdl_image_step = 0
        vdl_image_frame = 0  # each frame can display ten pictures at most.

    imid2path = dataset.get_imid2path()
    #print(imid2path[0])
    for iter_id, data in enumerate(loader()):
        try:
            outs = exe.run(infer_prog,
                           feed=data,
                           fetch_list=values,
                           return_numpy=False)
        except:
            print('detection error occur!')
            continue

        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))

        bbox_results = None
        mask_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        if 'mask' in res:
            mask_results = mask2out([res], clsid2catid,
                                    model.mask_head.resolution)

        #create save dir
        video_name = FLAGS.infer_dir.split("/")[-1]
        save_dir = os.path.join(FLAGS.output_dir, video_name)

        do_vis = False
        # visualize result
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            print(image_path)
            image = Image.open(image_path).convert('RGB')
            export_results_to_txt(image_path, bbox_results, FLAGS.output_dir)

            if do_vis:
                os.makedirs(save_dir, exist_ok=True)
                # use VisualDL to log original image
                if FLAGS.use_vdl:
                    original_image_np = np.array(image)
                    vdl_writer.add_image(
                        "original/frame_{}".format(vdl_image_frame),
                        original_image_np,
                        vdl_image_step)

                image = visualize_results(image,
                                          int(im_id), catid2name,
                                          FLAGS.draw_threshold, bbox_results,
                                          mask_results)

                # use VisualDL to log image with bbox
                if FLAGS.use_vdl:
                    infer_image_np = np.array(image)
                    vdl_writer.add_image(
                        "bbox/frame_{}".format(vdl_image_frame),
                        infer_image_np,
                        vdl_image_step)
                    vdl_image_step += 1
                    if vdl_image_step % 10 == 0:
                        vdl_image_step = 0
                        vdl_image_frame += 1

                save_name = get_save_image_name(save_dir, image_path)
                logger.info("Detection bbox results save in {}".format(save_name))
                image.save(save_name, quality=95)

    if FLAGS.use_vdl:
        vdl_writer.close()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    FLAGS = parser.parse_args()
    main()
