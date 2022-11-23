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
    parser.add_argument('--name', type = str, default = 'flower102')
    parser.add_argument('--train_dir', default='../CoTuning/data/finetune/flower102/train')
    parser.add_argument('--eval_dir', default='../CoTuning/data/finetune/flower102/test')
    parser.add_argument('--log_dir', default = './visual_log')
    parser.add_argument('--save', type = str, default = './output')
    parser.add_argument('--ema_decay', type = float, default = 0.999)
    parser.add_argument('--model_arch', default='resnet50')
    parser.add_argument('--image_size', type = int, default = 224)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--max_iters', type=int, default=9000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--alpha', type = float, default = 0.2, help = 'coefficient of mixup')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_frequency', type=int, default=50)
    parser.add_argument('--eval_frequency', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--reg_lambda', type=float, default=0.01)
    parser.add_argument('--aux_lambda', type=float, default=0.1)
    parser.add_argument('--cls_lambda', type=float, default=0.0001)
    parser.add_argument('--regularizer', type = str, default = 'smile')
    
    args = parser.parse_args()
    return args


def get_dataloader_train(args):
    train_path = args.train_dir
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = DatasetFolder(train_path, transform=transform_train)
    train_loader = paddle.io.DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    num_classes = len(train_set.classes)

    return train_loader, num_classes


def get_dataloader_val(args):
    val_path = args.eval_dir
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_set = DatasetFolder(val_path, transform=transform_val)
    val_loader = paddle.io.DataLoader(val_set, shuffle=False, batch_size=args.batch_size_eval)

    return val_loader



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



def mixup_data(x, y, index=None, alpha=0.2):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    if lam < 0.5:
        lam = 1 - lam
    batch_size = x.shape[0]
    if index is None:
        index = paddle.randperm(batch_size).numpy()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion_hard(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

logsoftmax = paddle.nn.LogSoftmax(axis=1)
def mixup_criterion_soft(pred, y_a, y_b, lam):
    log_probs = logsoftmax(pred)
    loss_a = (-y_a * log_probs).mean(0).sum()   
    loss_b = (-y_b * log_probs).mean(0).sum()
    loss = lam * loss_a + (1 - lam) * loss_b   
    return loss

def feature_interpolation(fm_src, fm_tgt, lam, index_perb):
    fm_src = fm_src.detach()
    b, c, h, w = fm_src.shape
    fm_src = lam * fm_src + (1 - lam) * fm_src[index_perb]
    fea_loss = paddle.norm(fm_src - fm_tgt) / (h * w)
    return fea_loss


def reg_fc(model):
    l2_cls = 0
    for name, param in model.named_parameters():
        if name.startswith('fc.') or name.startswith('aux_fc.'):
            l2_cls += 0.5 * paddle.norm(param) ** 2
    return l2_cls

def update_mean_teacher(ema_decay, model_source, model_tgt): # debug
    alpha = ema_decay
    #alpha = min(1 - 1 / (args.max_iters + 1), args.ema_decay)
    new_dict = {}
    for name, src_param in model_source.named_parameters():
        if name.startswith('fc.'):
            new_dict[name] = src_param
            continue
        tgt_param = model_tgt.state_dict()[name]
        src_param = alpha * src_param + (1 - alpha) * tgt_param
        new_dict[name] = src_param
        # src_param.data.mul_(alpha).add_(1 - alpha, tgt_param.data) 
        # model_source.state_dict()[name].set_dict(alpha * src_param + (1 - alpha) * tgt_param)
    model_source.set_dict(new_dict)
    
    



def train(iter_tgt,
          model_source,
          model_tgt,
          reg_lambda,
          aux_lambda,
          cls_lambda,
          alpha,
          criterion,
          ema_decay,
          optimizer,
          cur_iter,
          total_iter,
          debug_steps=100,
          logger=None,
          cur_regularizer='smile'):
    
    
    model_tgt.train()
    time_st = time.time()
    
    data = iter_tgt.next()
    image = data[0]
    label = paddle.unsqueeze(data[1], 1)
    
    if cur_regularizer == 'smile':
        # mix up
        index_perm = paddle.randperm(image.shape[0]).numpy()
        inputs_mix, targets_a, targets_b, lam = mixup_data(image, label, index=index_perm, alpha=alpha)
        logits_mix, features_mix, outputs_aux = model_tgt(inputs_mix)
        loss_main = mixup_criterion_hard(criterion, logits_mix, targets_a, targets_b, lam)
        loss_all = {'loss_main': loss_main}
    
        logits_src, feature_scr = model_source(image)
        outputs_src = F.softmax(logits_src, axis=1)
        loss_aux = mixup_criterion_soft(outputs_aux, outputs_src, outputs_src[index_perm], lam)
        loss_all['loss_aux'] = aux_lambda * loss_aux
        
        loss_reg = feature_interpolation(feature_scr, features_mix, lam, index_perm)
        loss_all['loss_reg'] = reg_lambda * loss_reg
        
        if ema_decay < 1-1e-6 and cur_iter % 10 == 0:
            update_mean_teacher(ema_decay, model_source, model_tgt)
    elif cur_regularizer == 'l2':
        logits, _, _ = model_tgt(image)
        loss_main = criterion(logits, label)
        loss_all = {'loss_main': loss_main}
    loss_classifier = reg_fc(model_tgt)
    loss_all['loss_classifier'] = cls_lambda * loss_classifier
    
    loss = sum(loss_all.values())
        
    

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    
    model_tgt.eval()
    with paddle.no_grad():
        logits, _, _ =  model_tgt(image)
    model_tgt.train()
    acc = paddle.metric.accuracy(logits, label)
    train_time = time.time() - time_st
    if logger and cur_iter % debug_steps == 0:
        logger.info(
            f"Step[{cur_iter:04d}/{total_iter:04d}], " +
            f"Loss is: {loss.numpy()}, " +
            f"Loss all: {loss_all}" +
            f"Train ACC@1: {acc.numpy()}"+
            f"Train Time: {train_time}")
    return loss.numpy(), acc.numpy()
    


def validate(dataloader, model_tgt, criterion, total_batch, debug_steps=100, logger=None):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        val_acc5_meter.avg: float, average top5 accuracy on current process/gpu
        val_time: float, valitaion time
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

            if logger and batch_id % debug_steps == 0 and batch_id != 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {avg_loss}, " +
                    f"Avg Acc@1: {avg_acc}, ")

    val_time = time.time() - time_st
    return avg_loss, avg_acc, val_time


def finetune_cnn(args):
    # STEP 0: Preparation
    
    last_epoch = -1
    paddle.device.set_device(f'gpu:{args.gpu}')
    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    logger = get_logger(filename=os.path.join(args.save, f'{args.name}_{args.regularizer}.txt'))
    logdir = os.path.join(args.log_dir, args.regularizer)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = LogWriter(logdir = logdir)
    logger.info(f'\n{args}')

    # STEP 1: Create train and val dataloader
    dataloader_train, num_classes = get_dataloader_train(args)
    if os.path.exists(args.eval_dir):
        dataloader_val = get_dataloader_val(args)

    # STEP 2: load model
    model_source = eval(args.model_arch)(pretrained=True, num_classes = 1000) # imagenet pretrained
    model_tgt = eval(args.model_arch)(pretrained=True, num_classes = num_classes, aux_head=True) # setting aux classifier (aux head)
    model_tgt.aux_fc.set_dict(model_source.fc.state_dict()) # auc_fc use pretrained ckpt!
    logger.info('finish load the pretrained model')

    # STEP 3: freeze model_src
    # algo = determine_algo(model, args, dataloader_train)
    model_source.eval()
    for param in model_source.parameters():
        param.stop_gradient = True

    # STEP 4: Define optimizer and lr_scheduler
    criterion = paddle.nn.CrossEntropyLoss()
    params = model_tgt.parameters()
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[int(2.0*(args.max_iters+1000)/3.0)],values=[args.lr,args.lr*0.1])
    optimizer = paddle.optimizer.Momentum(learning_rate=lr_scheduler, parameters=params,momentum=0.9, use_nesterov=True, weight_decay = args.wd)

    # STEP 5: Run training
    logger.info(f"Start training from iter 0.")
    len_tgt = len(dataloader_train)
    iter_tgt = iter(dataloader_train)
    best_val_acc = 0.0
    for cur_iter in range(0, args.max_iters):
        # train
        cur_regularizer = args.regularizer
        if args.regularizer == 'smile' and cur_iter >= args.max_iters - 1000:
            cur_regularizer = 'l2'
        if cur_iter % 500 == 0:
            logger.info(f"Now training iter {cur_iter}. LR={optimizer.get_lr():.6f}")
        if (cur_iter + 1) % len_tgt == 0:
            iter_tgt = iter(dataloader_train)
        train_loss, train_acc = train(iter_tgt=iter_tgt,
                                        model_source=model_source,
                                        model_tgt=model_tgt,
                                        reg_lambda=args.reg_lambda,
                                        aux_lambda=args.aux_lambda,
                                        cls_lambda=args.cls_lambda,
                                        alpha=args.alpha,
                                        criterion=criterion,
                                        ema_decay=args.ema_decay,
                                        optimizer=optimizer,
                                        cur_iter=cur_iter,
                                        total_iter=args.max_iters,
                                        debug_steps=args.print_frequency,
                                        logger=logger,
                                        cur_regularizer=cur_regularizer)
        lr_scheduler.step()
        writer.add_scalar(tag="train_acc", step=cur_iter, value=train_acc)
        writer.add_scalar(tag="train_loss", step=cur_iter, value=train_loss)

        # validation and save ckpts
        if (cur_iter % args.eval_frequency == 0 and cur_iter != 0) or (cur_iter+1) == args.max_iters:
            logger.info(f'----- Validation after iter: {cur_iter}')
            val_loss, val_acc, val_time = validate(
                dataloader=dataloader_val,
                model_tgt=model_tgt,
                criterion=criterion,
                total_batch=len(dataloader_val),
                debug_steps=args.print_frequency,
                logger=logger)
            logger.info(f"----- Iter[{cur_iter:03d}/{args.max_iters:03d}], " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Acc@1: {val_acc:.4f}, " +
                        f"time: {val_time:.2f}")
            writer.add_scalar(tag="val_acc", step=cur_iter, value=val_acc)
            writer.add_scalar(tag="val_loss", step=cur_iter, value=val_loss)
            
            # save if necessary
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(args.save, f"{args.name}_{args.regularizer}_Best.pdparams")
                state_dict = dict()
                state_dict['model'] = model_tgt.state_dict()
                state_dict['optimizer'] = optimizer.state_dict()
                state_dict['iter'] = cur_iter
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                paddle.save(state_dict, model_path)
                logger.info(f"----- Save model: {model_path}")
            print('Current best acc on val set is: ', best_val_acc)


if __name__ == '__main__':
    print(paddle.__version__)
    args = get_args()
    finetune_cnn(args)
    
