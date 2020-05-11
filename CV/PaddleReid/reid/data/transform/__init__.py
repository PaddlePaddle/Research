from . import operators

from .operators import *
import pdb

def build_transform(cfg):
    if cfg.use_autoaug:
        train_transform = [DecodeImage(with_mixup=True),
                       ResizeRandomCrop(big_height=cfg.big_height, big_width=cfg.big_width, target_height=cfg.target_height, target_width=cfg.target_width),
                       RandomHorizontalFlip(),
                       ImageNetPolicy(),
                       NormalizeImage(),
                       RandomErasing(cfg.re_prob),
                       Permute()]
    else:
        train_transform = [DecodeImage(),
                       ResizeRandomCrop(big_height=cfg.big_height, big_width=cfg.big_width,target_height=cfg.target_height, target_width=cfg.target_width),
                       RandomHorizontalFlip(),
                       NormalizeImage(),
                       RandomErasing(cfg.re_prob),
                       Permute()]
    if cfg.use_flip:
        test_transform = [DecodeImage(),
                      ResizeImage(height=cfg.target_height, width=cfg.target_width), 
                      HorizontalFlip(),
                      NormalizeImage(),
                      Permute()]
    else:
        test_transform = [DecodeImage(),
                      ResizeImage(height=cfg.target_height, width=cfg.target_width), 
                      NormalizeImage(),
                      Permute()]

    return train_transform, test_transform