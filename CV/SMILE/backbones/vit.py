# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
"""
ViT in Paddle
A Paddle Implementation of Vision Transformer (ViT) as described in:
"An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale"
    - Paper Link: https://arxiv.org/abs/2010.11929
"""
import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    This layer is used to avoid using 'if' condition in methods such as forward
    """
    def forward(self, x):
        return x


class PatchEmbedding(nn.Layer):
    """Patch Embedding
    Apply patch embedding (which is implemented using Conv2D) on input data.
    Attributes:
        image_size: image size
        patch_size: patch size
        num_patches: num of patches
        patch_embddings: patch embed operation (Conv2D)
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)  # [B, C, H, W] -> [B, C, h*w]
        x = x.transpose([0, 2, 1])  # [B, C, h*w] -> [B, h*w, C] = [B, N, C]
        return x


class Attention(nn.Layer):
    """ Attention module
    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        num_heads: number of heads
        attn_head_size: feature dim of single head
        all_head_size: feature dim of all heads
        qkv: a nn.Linear for q, k, v mapping
        scales: 1 / sqrt(single_head_feature_dim)
        out: projection of multi-head attention
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if attn_head_size is not None:
            self.attn_head_size = attn_head_size
        else:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3,  # weights for q, k, and v
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(self.all_head_size,
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        """[B, N, C] -> [B, N, n_heads, head_dim] -> [B, n_heads, N, head_dim]"""
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)  # [B, N, C] -> [B, N, n_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim] -> [B, n_heads, N, head_dim]
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)  # [B, n_heads, N, N]
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)  # [B, n_heads, N, head_dim]
        z = z.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim]
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)  # [B, N, all_head_size]

        z = self.out(z)
        z = self.proj_dropout(z)
        return z


class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc
    """

    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=0.2))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Layer):
    """Transformer Layer
    Transformer layer contains attention, norm, mlp and residual
    Attributes:
        embed_dim: transformer feature dim
        attn_norm: nn.LayerNorm before attention
        mlp_norm: nn.LayerNorm before mlp
        mlp: mlp modual
        attn: attention modual
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.attn_norm = nn.LayerNorm(embed_dim,
                                      weight_attr=w_attr_1,
                                      bias_attr=b_attr_1,
                                      epsilon=1e-6)

        self.attn = Attention(embed_dim,
                              num_heads,
                              attn_head_size,
                              qkv_bias,
                              dropout,
                              attention_dropout)

        #self.drop_path = DropPath(droppath) if droppath > 0. else Identity()

        w_attr_2, b_attr_2 = self._init_weights()
        self.mlp_norm = nn.LayerNorm(embed_dim,
                                     weight_attr=w_attr_2,
                                     bias_attr=b_attr_2,
                                     epsilon=1e-6)

        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        #x = self.drop_path(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        #x = self.drop_path(x)
        x = x + h

        return x


class Encoder(nn.Layer):
    """Transformer encoder
    Encoder encoder contains a list of TransformerLayer, and a LayerNorm.
    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        # stochatic depth decay
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(TransformerLayer(embed_dim,
                                               num_heads,
                                               attn_head_size,
                                               qkv_bias,
                                               mlp_ratio,
                                               dropout,
                                               attention_dropout,
                                               depth_decay[i]))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights()
        self.encoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return x


class VisionTransformer(nn.Layer):
    """ViT transformer
    ViT Transformer, classifier is a single Linear layer for finetune,
    For training from scratch, two layer mlp should be used.
    Classification is done using cls_token.
    Args:
        image_size: int, input image size, default: 224
        patch_size: int, patch size, default: 16
        in_channels: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        embed_dim: int, embedding dimension (patch embed out dim), default: 768
        depth: int, number ot transformer blocks, default: 12
        num_heads: int, number of attention heads, default: 12
        attn_head_size: int, dim of head, if none, set to embed_dim // num_heads, default: None
        mlp_ratio: float, ratio of mlp hidden dim to embed dim(mlp in dim), default: 4.0
        qkv_bias: bool, If True, enable qkv(nn.Linear) layer with bias, default: True
        dropout: float, dropout rate for linear layers, default: 0.
        attention_dropout: float, dropout rate for attention layers default: 0.
        droppath: float, droppath rate for droppath layers, default: 0.
        representation_size: int, set representation layer (pre-logits) if set, default: None
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 attn_head_size=None,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 representation_size=None):
        super().__init__()
        # create patch embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              embed_dim)
        # create posision embedding
        self.position_embedding = paddle.create_parameter(
            shape=[1, 1 + self.patch_embedding.num_patches, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        # create cls token
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        self.pos_dropout = nn.Dropout(dropout)
        # create multi head self-attention layers
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               attn_head_size,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath)
        # pre-logits
        if representation_size is not None:
            self.num_features = representation_size
            w_attr_1, b_attr_1 = self._init_weights()
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim,
                          representation_size,
                          weight_attr=w_attr_1,
                          bias_attr=b_attr_1),
                nn.ReLU())
        else:
            self.pre_logits = Identity()

        # classifier head
        w_attr_2, b_attr_2 = self._init_weights()
        self.classifier = nn.Linear(embed_dim,
                                    num_classes,
                                    weight_attr=w_attr_2,
                                    bias_attr=b_attr_2)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward_features(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.position_embedding
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.pre_logits(x[:, 0]) # cls_token only
        return x

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.classifier(x)
        return logits


def build_vit(config):
    """build vit model from config"""
    model = VisionTransformer(image_size=config.DATA.IMAGE_SIZE,
                              patch_size=config.MODEL.PATCH_SIZE,
                              in_channels=config.DATA.IMAGE_CHANNELS,
                              num_classes=config.MODEL.NUM_CLASSES,
                              embed_dim=config.MODEL.EMBED_DIM,
                              depth=config.MODEL.DEPTH,
                              num_heads=config.MODEL.NUM_HEADS,
                              attn_head_size=config.MODEL.ATTN_HEAD_SIZE,
                              mlp_ratio=config.MODEL.MLP_RATIO,
                              qkv_bias=config.MODEL.QKV_BIAS,
                              dropout=config.MODEL.DROPOUT,
                              attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                              droppath=config.MODEL.DROPPATH,
                              representation_size=None)
    return model