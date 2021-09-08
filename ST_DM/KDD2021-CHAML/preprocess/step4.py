import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import x2paddle.torch2paddle as init
import pickle
import numpy as np
import os
import time
from constants import *
"""
step4: generate the POI id embedding files for each city. These embeddings are fixed during training.
TODO: according to the paper, you may pre-train the POI id embeddings by NeuMF model on the data
      collected before the dataset time span. Here we use random embeddings for simplicity.
"""


class NullEmbedder(nn.Layer):

    def __init__(self, poi_size, embed_dim):
        super(NullEmbedder, self).__init__()
        self.poi_embedding = nn.Embedding(poi_size, embed_dim, padding_idx=0, weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()))
        # init.xavier_uniform_(self.poi_embedding.weight)

    def forward(self, save_file):
        w = self.poi_embedding.weight.detach().numpy()
        np.save(save_file, w)


def get_null_id_emb(city):
    EMBED_DIM = 50
    poiid_pkl_file = '{}{}/poiid_to_id.pkl'.format(pkl_path, city)
    poiid_pkl = pickle.load(open(poiid_pkl_file, 'rb'))
    poi_size = len(poiid_pkl)
    null_embedder = NullEmbedder(poi_size, EMBED_DIM)
    save_file = '{}{}_poiid_embed.npy'.format(save_path, city)
    null_embedder.forward(save_file)


if __name__ == '__main__':
    pkl_path = root_path + 'pkls/'
    save_path = root_path + 'id_embs/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cities = get_cities('base') + get_cities('valid') + get_cities('target')
    for city in cities:
        get_null_id_emb(city)
