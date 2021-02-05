#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: nets/common/losses.py
Author: map(zhuoan@baidu.com)
Date: 2020/06/15 12:14:18
"""

import math
import numpy as np
import logging
import collections
import paddle.fluid as fluid


def safe_cosine_sim(x, y):
    """
        fluid.layers.cos_sim maybe nan
        avoid nan
    """
    l2x = fluid.layers.l2_normalize(x, axis=-1)
    l2y = fluid.layers.l2_normalize(y, axis=-1)
    cos = fluid.layers.reduce_sum(l2x * l2y, dim=1, keep_dim=True)
    return cos


def loss_neg_log_of_pos(pos_score, neg_score_n, gama=5.0):
    """
        pos_score: batch_size x 1
        neg_score_n: batch_size x n
    """
    # n x batch_size
    neg_score_n = fluid.layers.transpose(neg_score_n, [1, 0])
    # 1 x batch_size
    pos_score = fluid.layers.reshape(pos_score, [1, -1])

    exp_pos_score = fluid.layers.exp(pos_score * gama)
    exp_neg_score_n = fluid.layers.exp(neg_score_n * gama)

    ## (n+1) x batch_size
    pos_neg_score = fluid.layers.concat([exp_pos_score, exp_neg_score_n], axis=0)
    ## 1 x batch_size
    exp_sum = fluid.layers.reduce_sum(pos_neg_score, dim=0, keep_dim=True)
    ## 1 x batch_size
    loss = -1.0 * fluid.layers.log(exp_pos_score / exp_sum)
    # batch_size
    loss = fluid.layers.reshape(loss, [-1, 1])
    #return [loss, exp_pos_score, exp_neg_score_n, pos_neg_score, exp_sum]
    return loss


def loss_pairwise_hinge(pos, neg, margin=0.8):
    """
        pairwise
    """
    loss_part1 = fluid.layers.elementwise_sub(
        fluid.layers.fill_constant_batch_size_like(
            input=pos, shape=[-1, 1], value=margin, dtype='float32'), pos)
    loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
    loss_part3 = fluid.layers.elementwise_max(
        fluid.layers.fill_constant_batch_size_like(
            input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'), loss_part2)
    return loss_part3


def simnet_circle_loss(sp, sn, margin, scale):
    """
    sp: score list of positive samples, shape [B * m]
    sn: score list of negative samples, shape [B * n]
    margin: relaxation factor in circle loss function
    scale:  scale factor in circle loss function

    return: circle loss value, shape [1]
    """
    op = 1. + margin
    on = 0. - margin
    delta_p = 1 - margin
    delta_n = margin

    ap = fluid.layers.relu(fluid.layers.elementwise_sub(sp, op) * -1.0)
    ap.stop_gradient =True
    an = fluid.layers.relu(fluid.layers.elementwise_sub(sn, on))
    an.stop_gradient =True

    logit_p = ap * (sp - delta_p)
    logit_p = logit_p * scale * -1.0
    logit_p = fluid.layers.cast(x=logit_p, dtype=np.float64)
    loss_p = fluid.layers.reduce_sum(fluid.layers.exp(logit_p), dim=1, keep_dim=False)

    logit_n = an * (sn - delta_n)
    logit_n = logit_n * scale
    logit_n = fluid.layers.cast(x=logit_n, dtype=np.float64)
    loss_n = fluid.layers.reduce_sum(fluid.layers.exp(logit_n), dim=1, keep_dim=False)

    circle_loss = fluid.layers.log(1 + loss_n * loss_p)
    circle_loss = fluid.layers.cast(x=circle_loss, dtype=np.float32)
    return fluid.layers.mean(circle_loss)


def ppl_eval(label_len, loss):
    """ppl"""

    cost_train = np.mean(loss)

    if isinstance(label_len, list):
        tmp_arr = []
        for one_batch in label_len:
            batch_arr = [one_len for one_len in one_batch]
            tmp_arr.extend(batch_arr)
        label_len = np.array(tmp_arr)
        
    word_count = np.sum(label_len)
    total_loss = cost_train * label_len.shape[0]
    try_loss = total_loss / word_count
    ppl = np.exp(total_loss / word_count)

    result = {"ave_loss": float("%.4f" % try_loss), "ppl": int(ppl)}
    return result


def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i: i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / (ratio + 1e-4))

    bleu = geo_mean * bp
    ret = [bleu, precisions, bp, ratio, translation_length, reference_length]
    return ret


def scale_l2(x, norm_length):
    """
    # copy lines from here https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py#L190
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    """

    alpha = fluid.layers.reduce_max(fluid.layers.abs(x), dim=1, keep_dim=True) + 1e-12
    l2_norm = alpha * fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.pow(x / alpha), dim=1, keep_dim=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def pgd_loss(ernie, labels, loss, task_fc_fn, epsilon=0.25):
    """ refer code from
    https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py#L145
    but we didn't use the vat loss for now
    """

    #TODO any difference with fleet_main_program or ParallelProgram or TrainProgram?
    program = fluid.default_main_program()

    param_grads = fluid.backward.append_backward(loss, parameter_list=[ernie._word_emb_name])

    # in the VAT paper code, the d is draw from a norm distribution, what's the advantage? why not use the
    # gradient of the emb directly?
    # d = fluid.layers.random_normal(shape=emb.shape)
    d = filter(lambda p: p[0].name == ernie._word_emb_name, param_grads)[0][1]
    emb = program.block(0).var(ernie._word_emb_name)

    #for _ in range(args.K_iteration):
    K_iteration = 8
    small_constant_for_finite_diff = 1e-5
    emb_hat = emb

    d = fluid.layers.gaussian_random(emb.shape)

    # it seems it can be implemented by the while loop
    for _ in range(K_iteration):
        #d = xi * utils_tf.l2_batch_normalize(d)
        d = scale_l2(d, small_constant_for_finite_diff)
        #logits_d = model.get_logits(x + d)
        #kl = utils_tf.kl_with_logits(logits, logits_d)

        emb_hat = emb_hat + d
        ernie._build_model(emb=emb_hat)
        graph_vars = task_fc_fn(ernie, labels)

        gradient = filter(lambda p: p[0].name == ernie._word_emb_name, param_grads)[0][1]
        gradient.stop_gradient = True
        d = gradient
        #Hd = tf.gradients(kl, d)[0]
        #d = tf.stop_gradient(Hd)

    d = scale_l2(d, small_constant_for_finite_diff)
    emb_hat = emb_hat + d
    ernie._build_model(emb=emb_hat)
    graph_vars = task_fc_fn(ernie, labels)

    return graph_vars['loss']


