#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import paddle


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         gather_idx=None,
                         param_initializer=None,
                         name='multi_head_att',
                         trainable=True):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = paddle.static.nn.fc(x=queries,
                                size=d_key * n_head,
                                num_flatten_dims=2,
                                weight_attr=paddle.ParamAttr(
                                    name=name + '_query_fc.w_0',
                                    trainable=trainable,
                                    initializer=param_initializer),
                                bias_attr=paddle.ParamAttr(
                                    name=name + '_query_fc.b_0',
                                    trainable=trainable))
        k = paddle.static.nn.fc(x=keys,
                                size=d_key * n_head,
                                num_flatten_dims=2,
                                weight_attr=paddle.ParamAttr(
                                    name=name + '_key_fc.w_0',
                                    trainable=trainable,
                                    initializer=param_initializer),
                                bias_attr=paddle.ParamAttr(
                                    name=name + '_key_fc.b_0',
                                    trainable=trainable))
        v = paddle.static.nn.fc(x=values,
                                size=d_value * n_head,
                                num_flatten_dims=2,
                                weight_attr=paddle.ParamAttr(
                                    name=name + '_value_fc.w_0',
                                    trainable=trainable,
                                    initializer=param_initializer),
                                bias_attr=paddle.ParamAttr(
                                    name=name + '_value_fc.b_0',
                                    trainable=trainable))
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = paddle.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head])

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return paddle.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]])

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = paddle.scale(x=q, scale=d_key ** -0.5)
        product = paddle.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = paddle.nn.functional.softmax(x=product)
        if dropout_rate:
            weights = paddle.fluid.layers.nn.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = paddle.matmul(x=weights, y=v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.

        cache_k, cache_v = cache["k"], cache["v"]
        select_k = paddle.gather(x=cache_k, index=gather_idx)
        select_v = paddle.gather(x=cache_v, index=gather_idx)
        select_k = paddle.reshape(x=select_k, shape=[0, 0, d_model])
        select_v = paddle.reshape(x=select_v, shape=[0, 0, d_model])
        k = paddle.concat(x=[select_k, k], axis=1)
        v = paddle.concat(x=[select_v, v], axis=1)
        paddle.assign(k, cache["k"])
        paddle.assign(v, cache["v"])

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = paddle.static.nn.fc(x=out,
                                   size=d_model,
                                   num_flatten_dims=2,
                                   weight_attr=paddle.ParamAttr(
                                       name=name + '_output_fc.w_0',
                                       trainable=trainable,
                                       initializer=param_initializer),
                                   bias_attr=paddle.ParamAttr(
                                       name=name + '_output_fc.b_0',
                                       trainable=trainable))
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn',
                              trainable=True):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = paddle.static.nn.fc(x=x,
                                 size=d_inner_hid,
                                 num_flatten_dims=2,
                                 activation=hidden_act,
                                 weight_attr=paddle.ParamAttr(
                                     name=name + '_fc_0.w_0',
                                     trainable=trainable,
                                     initializer=param_initializer),
                                 bias_attr=paddle.ParamAttr(
                                     name=name + '_fc_0.b_0',
                                     trainable=trainable))
    if dropout_rate:
        hidden = paddle.fluid.layers.nn.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)

    out = paddle.static.nn.fc(x=hidden,
                              size=d_hid,
                              num_flatten_dims=2,
                              weight_attr=paddle.ParamAttr(
                                  name=name + '_fc_1.w_0',
                                  trainable=trainable,
                                  initializer=param_initializer),
                              bias_attr=paddle.ParamAttr(
                                  name=name + '_fc_1.b_0',
                                  trainable=trainable))
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0., name='', trainable=True):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = paddle.static.nn.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=paddle.ParamAttr(
                    name=name + '_layer_norm_scale',
                    trainable=trainable,
                    initializer=paddle.nn.initializer.Constant(value=1.)),
                bias_attr=paddle.ParamAttr(
                    name=name + '_layer_norm_bias',
                    trainable=trainable,
                    initializer=paddle.nn.initializer.Constant(value=0.)))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = paddle.fluid.layers.nn.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(query_input,
                  key_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name='',
                  cache=None,
                  gather_idx=None,
                  trainable=True):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    key_input = pre_process_layer(
        key_input,
        preprocess_cmd,
        prepostprocess_dropout,
        name=name + '_pre_att',
        trainable=trainable) if key_input else None
    value_input = key_input if key_input else None

    attn_output = multi_head_attention(
        pre_process_layer(
            query_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att',
            trainable=trainable),
        key_input,
        value_input,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att',
        cache=cache,
        gather_idx=gather_idx,
        trainable=trainable)
    attn_output = post_process_layer(
        query_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att',
        trainable=trainable)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn',
            trainable=trainable),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn',
        trainable=trainable)
    return (post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn',
        trainable=trainable), ffd_output)


def encoder(enc_input,
            attn_bias=None,
            enc_layers=None,
            n_layer=None,
            n_head=None,
            d_key=None,
            d_value=None,
            d_model=None,
            d_inner_hid=None,
            prepostprocess_dropout=None,
            attention_dropout=None,
            relu_dropout=None,
            hidden_act=None,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name='',
            caches=None,
            gather_idx=None,
            trainable=True):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    if n_layer is None:
        assert enc_layers is not None
    else:
        assert enc_layers is None

    if enc_layers is None:
        enc_layers = list(range(n_layer))

    checkpoints = []
    for i in enc_layers:
        enc_output, cp = encoder_layer(
            enc_input,
            None,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i),
            cache=caches[i] if caches is not None else None,
            gather_idx=gather_idx,
            trainable=trainable)
        enc_input = enc_output
        checkpoints.append(cp.name)

    enc_output = pre_process_layer(
        enc_output, preprocess_cmd, prepostprocess_dropout, name="post_encoder", trainable=trainable)

    return enc_output, checkpoints
