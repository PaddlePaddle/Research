from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pickle
import numpy as np


class Learner(nn.Layer):

    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        p_type_size = self.config['num_poi_types']
        time_size = self.config['num_time']
        embed_dim = self.config['embed_dim']
        poiid_dim = self.config['poiid_dim']
        mlp_hidden = self.config['mlp_hidden']
        self.with_cont_feat = self.config['with_cont_feat']
        self.vars = nn.ParameterList()
        self.init_emb(1, poiid_dim)
        self.init_emb(p_type_size, embed_dim)
        self.init_emb(time_size, embed_dim)
        if self.with_cont_feat:
            candi_dim = poiid_dim + embed_dim * 2 + 2
        else:
            candi_dim = poiid_dim + embed_dim * 2
        self.init_fc(candi_dim * 4, embed_dim)
        self.init_fc(embed_dim, 1)
        self.init_fc(candi_dim * 2, mlp_hidden)
        self.init_fc(mlp_hidden, mlp_hidden)
        self.init_fc(mlp_hidden, 2)
        for i in range(self.config['global_fix_var']):
            self.vars[i].trainable = False

    def init_emb(self, max_size, embed_dim):
        w = paddle.create_parameter(shape=paddle.ones([max_size, embed_dim]
            ).requires_grad_(False).shape, dtype=str(paddle.ones([max_size,
            embed_dim]).requires_grad_(False).numpy().dtype),
            default_initializer=paddle.nn.initializer.XavierUniform())
        w.stop_gradient = False
        self.vars.append(w)

    def init_fc(self, input_dim, output_dim):
        w = paddle.create_parameter(shape=paddle.ones([input_dim,
            output_dim]).requires_grad_(False).shape, dtype=str(paddle.ones(
            [input_dim, output_dim]).requires_grad_(False).numpy().dtype),
            default_initializer=paddle.nn.initializer.XavierUniform())
        w.stop_gradient = False
        b = paddle.create_parameter(shape=paddle.zeros([output_dim]).
            requires_grad_(False).shape, dtype=str(paddle.zeros([output_dim])
            .requires_grad_(False).numpy().dtype), default_initializer=\
            paddle.nn.initializer.Assign(paddle.zeros([output_dim]).
            requires_grad_(False)))
        b.stop_gradient = False
        self.vars.append(w)
        self.vars.append(b)

    def attention(self, att_w1, att_b1, att_w2, att_b2, K, V, mask=None):
        """
        :param K: (batch_size, d)
        :param V: (batch_size, hist_len, d)
        :return: (batch_size, d)
        """
        K = paddle.expand(K.unsqueeze(axis=1), shape=V.size())
        fusion = paddle.concat([K, V, K - V, K * V], axis=-1)
        x = F.linear(fusion, att_w1, att_b1)
        x = F.relu(x)
        score = F.linear(x, att_w2, att_b2)
        if mask is not None:
            mask = mask.unsqueeze(axis=-1)
            wall = paddle.ones_like(score) * (-2 ** 32 + 1)
            score = paddle.where(mask==1, wall, score)
        alpha = F.softmax(score, axis=1)
        alpha = F.dropout(alpha, p=0.5)
        att = (alpha * V).sum(dim=1)
        return att

    def forward(self, batch_uid, batch_hist, batch_candi, vars=None, scaler
        =None):
        """
        :param batch_uid: (bsz, )
        :param batch_hist: (bsz, time_step, 5)
        :param batch_candi: (bsz, 5)
        :return: 
        """
        if vars is None:
            vars = self.vars
        poi_emb_w, poi_type_emb_w, time_emb_w = vars[0], vars[1], vars[2]
        att_w1, att_b1, att_w2, att_b2 = vars[3], vars[4], vars[5], vars[6]
        mlp_w1, mlp_b1, mlp_w2, mlp_b2, mlp_w3, mlp_b3 = vars[7], vars[8], vars[9], vars[10], vars[11], vars[12]
        if self.with_cont_feat and scaler is not None:
            hist_feat = []
            candi_feat = []
            if 'dist' in scaler:  # datapoint[3:]: u-p dist, dtime, delta_dist
                mean_dist, std_dist = scaler['dist']
                mean_dtime, std_dtime = scaler['dtime']
                try:
                    hist_feat.append((batch_hist[:, :, 3].unsqueeze(-1).
                        float() - mean_dist) / std_dist)
                except:
                    print('====debug====')
                    print(batch_hist.shape)
                    print(batch_hist[:, :, 3])
                    exit(2)
                hist_feat.append((batch_hist[:, :, 4].unsqueeze(-1).float() -
                    mean_dtime) / std_dtime)
                candi_feat.append((batch_candi[:, 3].unsqueeze(-1).float() -
                    mean_dist) / std_dist)
                candi_feat.append((batch_candi[:, 4].unsqueeze(-1).float() -
                    mean_dtime) / std_dtime)
            hist_embed = paddle.concat([F.embedding(x=batch_hist[:, :,
                0], weight=poi_emb_w, padding_idx=0), F.embedding(x=\
                batch_hist[:, :, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(x=batch_hist[:, :, 2], weight=time_emb_w),
                paddle.concat(hist_feat, axis=-1)], axis=-1)
            candi_embed = paddle.concat([F.embedding(x=batch_candi[:,
                0], weight=poi_emb_w, padding_idx=0), F.embedding(x=\
                batch_candi[:, 1], weight=poi_type_emb_w, padding_idx=0), F
                .embedding(x=batch_candi[:, 2], weight=time_emb_w),
                paddle.concat(candi_feat, axis=-1)], axis=-1)
        else:
            hist_embed = paddle.concat([F.embedding(x=batch_hist[:, :,
                0], weight=poi_emb_w, padding_idx=0), F.embedding(x=\
                batch_hist[:, :, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(x=batch_hist[:, :, 2], weight=time_emb_w)], axis=-1)
            candi_embed = paddle.concat([F.embedding(x=batch_candi[:,
                0], weight=poi_emb_w, padding_idx=0), F.embedding(x=\
                batch_candi[:, 1], weight=poi_type_emb_w, padding_idx=0), F
                .embedding(x=batch_candi[:, 2], weight=time_emb_w)], axis=-1)
        mask = batch_hist[:, :, 0] == 0
        hist = self.attention(att_w1, att_b1, att_w2, att_b2, candi_embed,
            hist_embed, mask)
        embeds = paddle.concat([hist, candi_embed], axis=-1)
        embeds = F.dropout(embeds, p=0.5)
        fc1 = F.linear(embeds, mlp_w1, mlp_b1)
        fc1 = F.relu(fc1)
        fc1 = F.dropout(fc1, p=0.5)
        fc2 = F.linear(fc1, mlp_w2, mlp_b2)
        fc2 = F.relu(fc2)
        fc2 = F.dropout(fc2, p=0.5)
        prediction = F.linear(fc2, mlp_w3, mlp_b3)
        return prediction

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with paddle.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.z
        :return:
        """
        return self.vars
