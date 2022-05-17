# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################
"""
配置文件加载器
"""

import functools
import codecs
import os
from typing import Any, Dict, Generic

import paddle
import yaml

from easymia.libs import manager


class Config(object):
    '''
    Training configuration parsing. The only yaml/yml file is supported.

    The following hyper-parameters are available in the config file:
        batch_size: The number of samples per gpu.
        iters: The total training steps.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to paddleseg.datasets.
            For specific transforms, please refer to paddleseg.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently PaddleSeg only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        loss: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,
            where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of
            model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of
            loss type must be consistent with coef.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to paddleseg.models.
            For backbone, please refer to paddleseg.models.backbones.

    Args:
        path (str) : The path of config file, supports yaml format only.

    Examples:

        from paddleseg.cvlibs.config import Config

        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)

        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset

        # the argument of model should be parsed after dataset,
        # since the model builder uses some properties in dataset.
        model = cfg.model
        ...
    '''

    def __init__(self,
                 path: str,
                 learning_rate: float = None,
                 batch_size: Any = None,
                 iters: int = None,
                 resume_model: str = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None
        self._losses = None
        self._metrics = None
        self._infer_plugin = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate, batch_size=batch_size, iters=iters, resume_model=resume_model)

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: float = None,
               batch_size: Any = None,
               iters: int = None,
               resume_model: str = None):
        '''Update config'''
        if learning_rate:
            if 'lr_scheduler' in self.dic:
                self.dic['lr_scheduler']['learning_rate'] = learning_rate
            else:
                self.dic['learning_rate']['value'] = learning_rate

        if batch_size:
            if isinstance(batch_size, int):
                self.dic['train_batch_size'] = batch_size
                self.dic['infer_batch_size'] = batch_size
            elif isinstance(batch_size, (tuple, list)):
                assert len(batch_size) == 2, \
                    "batch_size contains at most 2 elements, got {}.".format(len(batch_size))
                train_batch_size, infer_batch_size = batch_size
                self.dic['train_batch_size'] = int(train_batch_size)
                self.dic['infer_batch_size'] = int(infer_batch_size)
            else:
                raise TypeError("batch_size except a int or [int, int], got {}".format(type(batch_size)))

        if iters:
            self.dic['iters'] = iters

        if resume_model:
            self.dic['model']['pretrained'] = resume_model

    @property
    def batch_size(self) -> int:
        return self.train_batch_size, self.infer_batch_size

    @property
    def train_batch_size(self) -> int:
        batch_size = self.dic.get('batch_size', [1, 1])
        if isinstance(batch_size, (tuple, list)):
            return batch_size[0]
        elif isinstance(batch_size, int):
            return batch_size
        else:
            raise TypeError("batch_size expect an int or [int, int], got {}.".format(batch_size))

    @property
    def infer_batch_size(self) -> int:
        batch_size = self.dic.get('batch_size', [1, 1])
        if isinstance(batch_size, (tuple, list)):
            return batch_size[1]
        elif isinstance(batch_size, int):
            return batch_size
        else:
            raise TypeError("batch_size expect an int or [int, int], got {}.".format(batch_size))

    @property
    def mode(self) -> int:
        mode = self.dic.get('mode')
        if not mode:
            raise RuntimeError('No mode specified in the configuration file.')
        return mode

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        if not iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            raise RuntimeError(
                'No `lr_scheduler` specified in the configuration file.')
        params = self.dic.get('lr_scheduler')

        lr_type = params.pop('type')
        # specific configs
        if lr_type == 'PolynomialDecay':
            params.setdefault('decay_steps', self.iters)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)
        elif lr_type == "WarmupCosine":
            params.setdefault('decay_steps', self.iters - params.get("warmup_steps", 10))
            params.setdefault('warmed_lr', params.get("learning_rate"))
        elif lr_type == "CosineCyclicRestart":
            params.setdefault('total_steps', self.iters)
        elif lr_type == "WarmupPoly":
            params.setdefault('decay_steps', self.iters - params.get("warmup_steps", 10))
            params.setdefault('warmed_lr', params.get("learning_rate"))

        # return lr_scheduler
        if lr_type in paddle.optimizer.lr.__all__:
            return getattr(paddle.optimizer.lr, lr_type)(**params)
        elif lr_type in manager.SCHEDULES.components_dict:
            return self._load_component(lr_type)(**params)
        else:
            raise RuntimeError('lr scheduler {} not supported yet.'.format(lr_type))

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        lr = self.lr_scheduler
        args = self.optimizer_args
        optimizer_type = args.pop('type')

        if optimizer_type == 'sgd':
            return paddle.optimizer.Momentum(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type == "adamw":
            return paddle.optimizer.AdamW(
                lr, parameters=self.model.parameters(), **args)
        else:
            raise RuntimeError('Only sgd and adam optimizer support.')

    @property
    def optimizer_args(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)

        return args

    @property
    def loss(self) -> dict:
        args = self.dic.get('loss', {}).copy()
        if 'types' in args and 'coef' in args:
            len_types = len(args['types'])
            len_coef = len(args['coef'])
            if len_types != len_coef:
                if len_types == 1:
                    args['types'] = args['types'] * len_coef
                else:
                    raise ValueError(
                        'The length of types should equal to coef or equal to 1 in loss config, but they are {} and {}.'
                        .format(len_types, len_coef))
        else:
            raise ValueError(
                'Loss config should contain keys of "types" and "coef"')

        if not self._losses:
            self._losses = dict()
            for key, val in args.items():
                if key == 'types':
                    self._losses['types'] = []
                    for item in args['types']:
                        self._losses['types'].append(self._load_object(item))
                else:
                    self._losses[key] = val
            if len(self._losses['coef']) != len(self._losses['types']):
                raise RuntimeError(
                    'The length of coef should equal to types in loss config: {} != {}.'
                    .format(
                        len(self._losses['coef']), len(self._losses['types'])))
        return self._losses

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')
        if not 'num_classes' in model_cfg:
            num_classes = None
            if self.train_dataset_config:
                if hasattr(self.train_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.train_dataset_class.NUM_CLASSES
                elif hasattr(self.train_dataset, 'num_classes'):
                    num_classes = self.train_dataset.num_classes
            elif self.val_dataset_config:
                if hasattr(self.val_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.val_dataset_class.NUM_CLASSES
                elif hasattr(self.val_dataset, 'num_classes'):
                    num_classes = self.val_dataset.num_classes

            if not num_classes:
                raise ValueError(
                    '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
                )

            model_cfg['num_classes'] = num_classes

        if not self._model:
            self._model = self._load_object({k: v for k, v in model_cfg.items() if k != "pretrained"})

            if "pretrained" in model_cfg and model_cfg['pretrained'] is not None:
                if os.path.exists(model_cfg['pretrained']):
                    para_state_dict = paddle.load(model_cfg['pretrained'])

                    model_state_dict = self._model.state_dict()
                    keys = model_state_dict.keys()
                    num_params_loaded = 0
                    for k in keys:
                        if k not in para_state_dict:
                            print("{} is not in pretrained model".format(k))
                        elif list(para_state_dict[k].shape) != list(
                                model_state_dict[k].shape):
                            print(
                                "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                                .format(k, para_state_dict[k].shape,
                                        model_state_dict[k].shape))
                        else:
                            model_state_dict[k] = para_state_dict[k]
                            num_params_loaded += 1
                    self._model.set_dict(model_state_dict)
                    print("There are {}/{} variables loaded into {} from {}.".format(
                        num_params_loaded, len(model_state_dict), self._model.__class__.__name__, model_cfg['pretrained']))
                else:
                    print("[WARN] pretrained model is not founded in {}.".format(model_cfg['pretrained']))

        return self._model

    @property
    def metrics(self) -> Any:
        args = self.dic.get('metric', {}).copy()

        if not self._metrics:
            self._metrics = list()
            for key, val in args.items():
                if key == 'types':
                    for item in args['types']:
                        self._metrics.append(self._load_object(item))

        return self._metrics

    @property
    def infer_plugin(self) -> Any:
        args = self.dic.get('infer_plugin', []).copy()

        if not self._infer_plugin:
            self._infer_plugin = list()
            for item in args:
                self._infer_plugin.append(self._load_object(item))
        return self._infer_plugin

    @property
    def train_dataset_config(self) -> Dict:
        return self.dic.get('train_task', {}).copy()

    @property
    def val_dataset_config(self) -> Dict:
        return self.dic.get('val_task', {}).copy()

    @property
    def test_dataset_config(self) -> Dict:
        return self.dic.get('test_task', {}).copy()

    @property
    def train_dataset_class(self) -> Generic:
        dataset_type = self.train_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def val_dataset_class(self) -> Generic:
        dataset_type = self.val_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def test_dataset_class(self) -> Generic:
        dataset_type = self.test_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        return self._load_object(_train_dataset)

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        return self._load_object(_val_dataset)

    @property
    def test_dataset(self) -> paddle.io.Dataset:
        _test_dataset = self.test_dataset_config
        if not _test_dataset:
            return None
        return self._load_object(_test_dataset)

    def _load_component(self, com_name):
        # 需要mode参数的模块
        com_list_mode = [
            manager.MODELS, manager.TRANSFORMS, manager.TASKS, manager.LOSSES,
        ]

        # 不需要mode参数的模块
        com_list_general = [
            manager.SCHEDULES, manager.METRICS, manager.DATASETS, manager.PLUGINS
        ]

        for com in com_list_mode:
            if com_name in com.components_dict:
                return functools.partial(com[com_name], mode=self.mode)
        for com in com_list_general:
            if com_name in com.components_dict:
                return com[com_name]
        else:
            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, cfg: dict) -> Any:
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))

        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val
        return component(**params)

    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self.dic)
