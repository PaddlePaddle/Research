#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: concat.py
func: 多模态特征通过self attention融合
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""
import paddle
import paddle.nn as nn


def vec2index(vec, cls_nums):
    """向量转化为索引矩阵
    """
    batch_num = vec.shape[0]
    batch_index = paddle.zeros(shape=(batch_num, cls_nums), dtype='int64')
    for i in range(batch_num):
        index = (vec[i] == 1).nonzero(as_tuple=True)
        #one_hot = torch.nn.functional.one_hot(index[0], cls_nums)
        nums_index = index[0].shape[0]
        a = paddle.add(index[0], paddle.ones(shape=[nums_index, 1], dtype='int64'))[:, 0]
        batch_index[i, 0:nums_index] = a

    return batch_index


class AttentionFuse(nn.Layer):
    """attention fuse
    Args:
        cls_nums: 词的类别数
        num_bottleneck：特征向量维度
        num_heads： self attention维度
    """
    def __init__(self, cls_nums, num_bottleneck, num_heads):
        super(AttentionFuse, self).__init__()
        self.img_transform = nn.Sequential(
                                nn.Linear(2048, 256), 
                                nn.LeakyReLU(0.1),
                                nn.Dropout(p=0.5),
                                )

        self.embedding = nn.Embedding(cls_nums + 1, num_bottleneck, padding_idx=0)

        self.multihead_attention1 = nn.MultiHeadAttention(embed_dim=num_bottleneck, num_heads=num_heads, dropout=0.5)
        self.multihead_attention2 = nn.MultiHeadAttention(embed_dim=num_bottleneck, num_heads=num_heads, dropout=0.5)

        self.batchnorm = nn.BatchNorm1D(num_bottleneck * 2)

        #self.dropout = nn.Dropout(0.5)

        self.cls_nums = cls_nums

    def forward(self, img, words):
        """forward
        Args:
            img: feature map, dims: [batch, w, h, c]
            words: word, dims: [batch, n] (n 为句子长度)
        """
        img = img.transpose([0, 2, 3, 1]) # (batch, 7, 7, 2048)
        s1, s2, s3, s4 = img.shape
        img = img.reshape([s1, s2 * s3, s4])
        img = self.img_transform(img)

        words = vec2index(words, self.cls_nums)
        words = paddle.cast(words, 'int64')

        words = self.embedding(words)

        words_feat = self.multihead_attention1(words, img, img)
  
        img_feat = self.multihead_attention2(img, words, words)

        img_feat = img_feat.mean(axis=1)
        words_feat = words_feat.mean(axis=1)

        x = paddle.concat([img_feat, words_feat], axis=-1)
        x = self.batchnorm(x)

        return x

if __name__ == "__main__":
    op = AttentionFuse(5, 256, 2)
    img = paddle.randn(shape=[2, 2048, 7, 7])
    words = paddle.to_tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 1]])
    x = op(img, words)

    print(x.shape)

