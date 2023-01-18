#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: trainer.py
func: 训练代码 
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/08/02
"""
import copy
import os
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from visualdl import LogWriter
from paddle.vision import transforms
import numpy as np

from mmflib.data.imggeo_dataset import ImgGeoDataset
from mmflib.data.signboard_dataset import SignboardDataset
from mmflib.data.brand_dataset import BrandDataset
from mmflib.data import preprocess
from mmflib.data.RandomSampler import DistributedRandomIdentitySampler
from mmflib.arch.arch import MMFModel
from mmflib.loss.combined_loss import CombinedLoss
from mmflib.utils.config import print_config
from mmflib.metric.topk import TopkAcc
from mmflib.metric.signboard_pr import SignboardPR
from mmflib.metric.topkmap import TopKMAP


class Trainer(object):
    """trainer
    """
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.output_dir = self.config['Global']['output_dir']
        print_config(config)
        self.device = paddle.set_device(self.config['Global']['device'])
        self.model = self.build_model()
        if self.config["Global"]["pretrained_model"] is not None:
            self.load_pretrain()
            print("Model restored from %s" % self.config["Global"]["pretrained_model"])
        else:
            print("[!] Retrain")
        self.model = paddle.DataParallel(self.model)

        self.vdl_writer = None
        if self.config["Global"]["use_visualdl"] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, 'vdl_log')
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)
        
        self.train_loader = None
        self.val_loader = None
        self.query_loader = None
        self.gallery_loader = None
        self.eval_dic = {} 

        self.train_loss_func = None
        self.train_metric_func = None
        self.eval_metric_func = None

        self.iters = 0
        self.best_metric_value = 0
        
    def train(self):
        """train"""
        if self.train_loss_func is None:
            loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = self.build_loss(loss_info)
        if self.train_metric_func is None:
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Train")
                if metric_config is not None:
                    self.train_metric_func = self.build_metrics(metric_config)
        if self.train_loader is None:
            self.train_loader = self.build_dataloader('Train')
        
        step_each_epoch = len(self.train_loader)
        print("STEP PER EACH EPOCH: ", step_each_epoch)

        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=self.config["Global"]["lr"], 
                                                             T_max=self.config["Global"]["epochs"], 
                                                             verbose=True)
        optimizer = paddle.optimizer.SGD(parameters=self.model.parameters(), 
                                         learning_rate=scheduler, 
                                         weight_decay=1e-4)

        if not os.path.exists("./output/checkpoints"):
            os.makedirs("./output/checkpoints")
        
        for epoch in range(1, self.config["Global"]["epochs"] + 1):
            self.model.train()
            pbar = tqdm(self.train_loader)
            for ret_data in pbar:
                img = ret_data['img']
                word= ret_data['word']
                if self.iters % 5000 == 0:
                    self.vdl_writer.add_image(tag="img", 
                                            img=(img * 0.5 + 0.5) * 255, 
                                            step=self.iters,
                                            dataformats="NCHW")
                labels = ret_data['label']
                img_path = ret_data['img_name']

                optimizer.clear_grad()

                ret_model = self.model(img, word, labels)

                ret_loss = self.train_loss_func(ret_model, labels)
                ret_metric = self.train_metric_func(ret_model, labels)

                for loss_name in ret_loss:
                    self.vdl_writer.add_scalar('Train: ' + loss_name, ret_loss[loss_name].numpy(), self.iters)
                for metric_name in ret_metric:
                    self.vdl_writer.add_scalar('Train: ' + metric_name, ret_metric[metric_name].numpy(), self.iters)
                
                ret_loss['loss'].backward()
                cur_lr = self.get_lr(optimizer)
                self.vdl_writer.add_scalar("LR", cur_lr, self.iters)
                optimizer.step()

                if self.iters % self.config["Global"]["save_interval"] == 0:
                    print("validation...")
                    metric_value = self.valid(epoch)
                    for metric_name in metric_value:
                        self.vdl_writer.add_scalar('Eval: ' + metric_name, metric_value[metric_name], self.iters)
                    if metric_value["metric_all"] > self.best_metric_value:
                        self.best_metric_value = metric_value["metric_all"]
                        self.save_ckpt('./output/checkpoints/best_model_{}_{}.pth'.format(epoch, self.iters))
                        print('EVAL BEST Metric: {}'.format(metric_value["metric_all"]))
                    else:
                        self.save_ckpt('./output/checkpoints/latest_model_{}_{}.pth'.format(epoch, self.iters))
                        print('EVAL LATEST Metric: {}'.format(metric_value["metric_all"]))

                    self.model.train()

                self.iters = self.iters + 1
                pbar.set_description("loss=%.5f, lr = %.6f" % (ret_loss['loss'].item(), cur_lr))
                
            scheduler.step()

    def valid(self, epochs):
        """valid
        Args:
            epochs: 当前迭代轮次
        Returns:
            metric_value: 当前验证轮次上的指标结果
        """
        if "Gallery" in self.config['DataLoader']:
            if self.gallery_loader is None:
                self.gallery_loader = self.build_dataloader('Gallery')
                self.eval_dic["gallery"] = self.gallery_loader
        if "Query" in self.config['DataLoader']:
            if self.query_loader is None:
                self.query_loader = self.build_dataloader('Query')
                self.eval_dic["query"] = self.query_loader
        if "Eval" in self.config['DataLoader']:
            if self.val_loader is None:
                self.val_loader = self.build_dataloader('Eval')  
                self.eval_dic["val"] = self.val_loader

        self.model.eval()
        with paddle.no_grad():
            for eval_name in self.eval_dic:
                with open(os.path.join(self.config["Global"]["output_dir"], eval_name + '.feature'), 'w') as f:
                    for i, data in enumerate(tqdm(self.eval_dic[eval_name])):
                        img = data['img']
                        word = data['word']
                        labels = data['label']
                        img_path = data['img_name']
                        now_batch_size, _, _, _ = img.shape

                        ret_model = self.model(img, word, labels)

                        embeddings = ret_model["features"]
                        embeddings_norm = ret_model["norm"]

                        for num in range(now_batch_size): 
                            content=[img_path[num]]
                            embedding=list(embeddings.detach().cpu()[num].numpy())
                            embedding=[str(x) for x in embedding]
                            content.extend([' '.join(embedding)])
                            x_norm = str(float(embeddings_norm.detach().cpu()[num]))
                            content.append(x_norm)
                            content='\t'.join(content)
                            f.write(content + '\n') 
                            
        metric_name = self.config["Metric"]["Eval"]["name"]
        metric_value = {}

        if metric_name == "SignboardPR":
            server_conf = {}
            server_conf['test_data_path'] = self.config["DataLoader"]["Eval"]["dataset"]["name"]
            server_conf['result_path'] = os.path.join(self.config["Global"]["output_dir"], "val.feature")
            server_conf['near_imid_path'] = self.config["Metric"]["Eval"]["near_imid"]
            server_conf['other_imid_path'] = self.config["Metric"]["Eval"]["other_imid"]
            server_conf['distance_limit'] = self.config["Metric"]["Eval"]["distance_limit"]
            server_conf['model_threshold'] = 1
            steps = '0'
            server_conf['output_file'] = './output/dist_{}_{}'.format(str(epochs), str(self.iters))

            self.eval_metric_func = SignboardPR()
            self.eval_metric_func.run2(server_conf, 256)

            p, r, top1, thr=self.eval_metric_func.metric('./output/dist_{}_{}'.format(str(epochs), 
                                                                   str(self.iters)), 15)
            with open('./output/metric_result', 'a') as f:
                f.write('Epoch:{}, STEP:{}, THRESHPLD: {}, RECALL: {}, PRECIOUS: {}, TOP1: {}\n'.format(str(epochs),
                                                                                                        str(self.iters),
                                                                                                        str(thr), 
                                                                                                        str(r), 
                                                                                                        str(p), 
                                                                                                        str(top1)))
            metric_value[metric_name] = 2 * p * r * 1.0 / (p + r)
            metric_value["metric_all"] = 2 * p * r * 1.0 / (p + r)

        elif metric_name == "TopKMAP":
            index_feature_path = os.path.join(self.config["Global"]["output_dir"], "gallery.feature")
            query_feature_path = os.path.join(self.config["Global"]["output_dir"], "query.feature")
            index_info_path = self.config["DataLoader"]["Gallery"]["dataset"]["cls_label_path"]
            query_info_path = self.config["DataLoader"]["Query"]["dataset"]["cls_label_path"]

            self.eval_metric_func = TopKMAP(index_info_path, 
                                            query_info_path, 
                                            query_feature_path, 
                                            index_feature_path)

            topkmap = self.eval_metric_func.calc_mapk_nn(self.config["Metric"]["Eval"]["topk"])

            metric_value[metric_name] = topkmap
            metric_value["metric_all"] = topkmap
        
        return metric_value      

    def build_model(self):
        """构建模型
        Returns:
           arch: 返回模型结构 
        """
        config = copy.deepcopy(self.config["Arch"])
        arc_name = config.pop("name")
        arch = eval(arc_name)(config)

        return arch

    def build_dataloader(self, mode):
        """构建数据迭代器
        Args:
            mode: 数据迭代器的类型 可选值 ['Train', 'Gallery', 'Query', 'Eval']
        Returns:
            data_loader: 返回对应类型的数据迭代器
        """
        assert mode in ['Train', 'Gallery', 'Query', 'Eval']
        config_dataset = self.config['DataLoader'][mode]['dataset']
        config_dataset = copy.deepcopy(config_dataset)

        config_transform = self.config['DataLoader'][mode]['transform_ops']
        transforms = self.build_transform(config_transform)

        dataset_name = config_dataset.pop('name')
        dataset = eval(dataset_name)(**config_dataset, transform_ops=transforms)

        config_sampler = self.config['DataLoader'][mode]['sampler']
        config_sampler = copy.deepcopy(config_sampler)
        if "name" not in config_sampler:
            batch_sampler = None
            batch_size = config_sampler["batch_size"]
        else:
            sampler_name = config_sampler.pop("name")
            batch_sampler = eval(sampler_name)(dataset.labels, **config_sampler)
            batch_size = config_sampler["batch_size"]
        
        config_loader = self.config['DataLoader'][mode]['loader']
        drop_last = config_loader['drop_last']
        num_workers = config_loader['num_workers']
        shuffle = config_loader["shuffle"]

        if batch_sampler is None:
            data_loader = DataLoader(dataset, 
                                     batch_size=batch_size, 
                                     places=self.device,
                                     shuffle=shuffle, 
                                     return_list=True,
                                     use_shared_memory=True,
                                     num_workers=num_workers, 
                                     drop_last=drop_last,
                                     batch_sampler=batch_sampler)
        else:
            data_loader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    places=self.device,
                                    shuffle=shuffle, 
                                    return_list=True,
                                    use_shared_memory=True,
                                    num_workers=num_workers, 
                                    drop_last=drop_last)

        return data_loader

    def build_transform(self, config_transform):
        """构建图像增强op
        Args:
           config_transform: 图像增强操作的配置参数，type: list 
        Returns:
            返回图像增强op
        """
        assert isinstance(config_transform, list), ('operator config should be a list')
        ops = []

        for operator in config_transform:
            assert isinstance(operator,
                            dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            op = getattr(preprocess, op_name)(**param)
            ops.append(op)

        return transforms.Compose(ops)
    
    def build_loss(self, loss_info):
        """构建损失函数
        Args:
           loss_info: 损失函数对应的配置参数 type: dict
        Returns:
            返回对应损失函数
        """
        config = copy.deepcopy(loss_info)
        return CombinedLoss(config)

    def build_metrics(self, config):
        """构建评估指标
        Args:
           config: 评估指标对应的配置参数 type: dict
        Returns:
            返回对应评估指标
        """
        config = copy.deepcopy(config)
        name = config.pop("name")
        return eval(name)(**config)

    def load_pretrain(self):
        """加载结构的预训练
        """
        checkpoint = paddle.load(self.config["Global"]["pretrained_model"])
        self.model.set_state_dict(checkpoint)

    def save_ckpt(self, path):
        """ save current model
        Args:
            path: 存储路径
        """
        paddle.save(self.model.state_dict(), path)
        print("Model saved as %s" % path)

    def get_lr(self, optimizer):
        """获取优化器的学习率
        Args:
            optimizer: 数据优化
        Returns:
            返回当前轮次优化器对应的学习率
        """
        return optimizer.get_lr()