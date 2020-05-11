from __future__ import absolute_import

from .resnet_vd import ResNet18_vd, ResNet34_vd, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd
from .resnext_vd import ResNeXt50_vd_64x4d, ResNeXt101_vd_64x4d, ResNeXt152_vd_64x4d, ResNeXt50_vd_32x4d, ResNeXt101_vd_32x4d, ResNeXt152_vd_32x4d
from .resnext_vd_multi_branch import ResNeXt101_vd_64x4d_MB, ResNeXt101_vd_32x4d_MB
from .se_resnet_vd import SE_ResNet18_vd, SE_ResNet34_vd, SE_ResNet50_vd, SE_ResNet101_vd, SE_ResNet152_vd, SE_ResNet200_vd
from .se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt101_vd_32x4d, SENet154_vd
from .resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl


from .feature_net import reid_feature_net
from .feature_net_multi_branch import reid_feature_net_mb

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr



__factory__ = {
    'ResNet101_vd': ResNet101_vd,                           
    
    'ResNeXt101_32x8d_wsl':ResNeXt101_32x8d_wsl,
    'ResNeXt101_32x16d_wsl':ResNeXt101_32x16d_wsl,          

    'ResNeXt101_vd_32x4d_MB':ResNeXt101_vd_32x4d_MB,
    'ResNeXt101_vd_64x4d_MB':ResNeXt101_vd_64x4d_MB,


}

class model_creator():
    def __init__(self, cfg):
        self.backbone = __factory__[cfg.model_arch]()
        if cfg.use_multi_branch:
            assert "MB" in cfg.model_arch
            self.feature_net = reid_feature_net_mb()
        else:
            self.feature_net = reid_feature_net()

    def net(self,input, is_train=False, class_dim=751, num_features = 512):
        backbone_feature = self.backbone.net(input=input, class_dim=class_dim)
        _, reid_fea = self.feature_net.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=class_dim)
        return reid_fea

    def net_multi_branch(self, input, is_train=True, class_dim=751, num_features=512):
        backbone_feature = self.backbone.net(input=input, class_dim=class_dim)
        if is_train:
            x3_g_pool_fc, x4_g_pool_fc, x4_p_pool_fc, x3_g_avg, x3_g_max, x4_g_avg, x4_g_max, x4_p_avg, x4_p_max = self.feature_net.net(input=backbone_feature, is_train=is_train, num_features=num_features, class_dim=class_dim)
            return x3_g_pool_fc, x4_g_pool_fc, x4_p_pool_fc, x3_g_avg, x3_g_max, x4_g_avg, x4_g_max, x4_p_avg, x4_p_max
        else:
            _, final_fea = self.feature_net.net(input=backbone_feature, is_train=is_train, num_features=num_features, class_dim=class_dim)
            return final_fea


    def net_pid_color_type(self,input, is_train=True, class_dim = 751, color_class = 12, type_class = 11, num_features = 512, finetune=False):
        backbone_feature = self.backbone.net(input=input)
        pid_cls, reid_feature = self.feature_net.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=class_dim, finetune=finetune)
        color_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=color_class)
        type_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=type_class)
        return pid_cls, color_cls, type_cls, reid_feature

