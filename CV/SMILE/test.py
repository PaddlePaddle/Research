import time
import logging
import os
import sys
import argparse
import random
from visualdl import LogWriter
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.vision import transforms
from paddle.vision.datasets import DatasetFolder

from backbones import mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152



def get_args():
    parser = argparse.ArgumentParser(description='PaddlePaddle Deep Transfer Learning Toolkit, Image Classification Fine-tuning Example')
    parser.add_argument('--test_dir', default='../CoTuning/data/finetune/flower102/test')
    parser.add_argument('--model_arch', default='resnet50')
    parser.add_argument('--ckpts', type = str)
    parser.add_argument('--image_size', type = int, default = 224)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args




def get_dataloader_test(args):
    test_path = args.test_dir
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_set = DatasetFolder(test_path, transform=transform_val)
    test_loader = paddle.io.DataLoader(test_set, shuffle=False, batch_size=args.batch_size_eval)
    num_classes = len(test_set.classes)

    return test_loader, num_classes



def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

    


def test(dataloader, model_tgt, criterion, total_batch, debug_steps=100):
    """Test for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        test_loss_meter.avg: float, average loss on current process/gpu
        test_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        test_time: float, test time
    """
    model_tgt.eval()
    losses = []
    accuracies = []
    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            label = paddle.unsqueeze(data[1], 1)
            logits, _, _= model_tgt(image)

            loss = criterion(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

            avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

            if batch_id % debug_steps == 0 and batch_id != 0:
                print(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {avg_loss}, " +
                    f"Avg Acc@1: {avg_acc}, ")

    val_time = time.time() - time_st
    return avg_loss, avg_acc, val_time


def test_cnn(args):
    # STEP 0: Preparation
    paddle.device.set_device(f'gpu:{args.gpu}')
    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # STEP 1: Create test dataloader
    dataloader_test, num_classes = get_dataloader_test(args)

    # STEP 2: Load model
    model_tgt = eval(args.model_arch)(pretrained=False, num_classes = num_classes, aux_head=True) # setting aux classifier (aux head)
    loaded_dict = paddle.load(args.ckpts)
    model_tgt.set_dict(loaded_dict['model']) # auc_fc use pretrained ckpt!
    print('finish load the finetuning model')

    # STEP 5: Testing
    criterion = paddle.nn.CrossEntropyLoss()
    print("Start testing...")
    
    val_loss, val_acc, val_time = test(
        dataloader=dataloader_test,
        model_tgt=model_tgt,
        criterion=criterion,
        total_batch=len(dataloader_test),
        debug_steps=50)
    print(f"Validation Loss: {val_loss:.4f}, " +
        f"Validation Acc@1: {val_acc:.4f}, " +
        f"time: {val_time:.2f}")
            


if __name__ == '__main__':
    print(paddle.__version__)
    args = get_args()
    test_cnn(args)
    
