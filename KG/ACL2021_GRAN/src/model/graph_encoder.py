#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Graph Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial, reduce

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper


def layer_norm(x,
               begin_norm_axis=1,
               epsilon=1e-12,
               param_attr=None,
               bias_attr=None):
    """
    Replace build-in layer_norm op with this function
    """
    helper = LayerHelper('layer_norm', **locals())
    mean = layers.reduce_mean(x, dim=begin_norm_axis, keep_dim=True)
    shift_x = layers.elementwise_sub(x=x, y=mean, axis=0)
    variance = layers.reduce_mean(
        layers.square(shift_x), dim=begin_norm_axis, keep_dim=True)
    r_stdev = layers.rsqrt(variance + epsilon)
    norm_x = layers.elementwise_mul(x=shift_x, y=r_stdev, axis=0)

    param_shape = [reduce(lambda x, y: x * y, norm_x.shape[begin_norm_axis:])]
    param_dtype = norm_x.dtype
    scale = helper.create_parameter(
        attr=param_attr,
        shape=param_shape,
        dtype=param_dtype,
        default_initializer=fluid.initializer.Constant(1.))
    bias = helper.create_parameter(
        attr=bias_attr,
        shape=param_shape,
        dtype=param_dtype,
        is_bias=True,
        default_initializer=fluid.initializer.Constant(0.))

    out = layers.elementwise_mul(x=norm_x, y=scale, axis=-1)
    out = layers.elementwise_add(x=out, y=bias, axis=-1)

    return out


def multi_head_attention(queries,
                         keys,
                         values,
                         edges_key,
                         edges_value,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head (Self) Attention Layer.
    This module calculates multi-head self-attention for a given labeled graph,
    which takes into account edges between input nodes. Note that attn_bias is
    added to logits before computing softmax activiation to mask certain selected
    positions so that they will not be considered in attention weights.
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
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permute the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def edge_aware_self_attention(q, k, v, edges_k, edges_v, attn_bias, d_key,
                                  dropout_rate):
        """
        Edge-aware Self-Attention.

        Scalar dimensions referenced here:
            B = batch_size
            M = max_sequence_length
            N = num_attention_heads
            H = hidden_size_per_head

        Args:
            q: reshaped queries [B, N, M, H]
            k: reshaped keys    [B, N, M, H]
            v: reshaped values  [B, N, M, H]
            edges_k: edge representations between input tokens (keys)   [M, M, H]
            edges_v: edge representations between input tokens (values) [M, M, H]
            attn_bias: attention mask [B, N, M, M]
        """
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise ValueError("Input q, k, v should be 4-D Tensors.")
        if not (len(edges_k.shape) == len(edges_v.shape) == 3):
            raise ValueError(
                "Input edges_k and edges_v should be 3-D Tensors.")

        # regular self-attention
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)

        # edge-aware self-attention
        if edges_k and edges_v:
            # 1. transpose scaled_q from [B, N, M, H] to [M, B, N, H]
            scaled_q = layers.transpose(x=scaled_q, perm=[2, 0, 1, 3])
            # 2. reshape scaled_q from [M, B, N, H] to [M, B*N, H]
            scaled_q = layers.reshape(
                x=scaled_q, shape=[0, -1, scaled_q.shape[3]], inplace=True)
            # 3. multiply scaled_q with transpose(edges_k)
            #      scaled_q:  [M, B*N, H]
            #      edges_k:   [M, M, H]
            #      edge_bias: [M, B*N, M]
            edge_bias = layers.matmul(x=scaled_q, y=edges_k, transpose_y=True)
            # 4. reshape edge_bias from [M, B*N, M] to [M, B, N, M]
            edge_bias = layers.reshape(
                x=edge_bias,
                shape=[0, -1, q.shape[1], q.shape[2]],
                inplace=True)
            # 5. transpose edge_bias from [M, B, N, M] to [B, N, M, M]
            edge_bias = layers.transpose(x=edge_bias, perm=[1, 2, 0, 3])
            # 6. add edge_bias to product
            product += edge_bias

        # add attention bias
        if attn_bias:
            product += attn_bias

        # softmax attention weights
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)

        # edge-aware self-attention
        out = layers.matmul(weights, v)
        if edges_k and edges_v:
            # 1. transpose weights from [B, N, M, M] to [M, B, N, M]
            reshaped_weights = layers.transpose(x=weights, perm=[2, 0, 1, 3])
            # 2. reshape weights from [M, B, N, M] to [M, B*N, M]
            reshaped_weights = layers.reshape(
                x=reshaped_weights,
                shape=[0, -1, reshaped_weights.shape[3]],
                inplace=True)
            # 3. multiply reshaped_weights with edges_v
            #      reshaped_weights: [M, B*N, M]
            #      edges_v:          [M, M, H]
            #      edge_bias:        [M, B*N, H]
            edge_bias = layers.matmul(x=reshaped_weights, y=edges_v)
            # 4. reshape edge_bias from [M, B*N, H] to [M, B, N, H]
            edge_bias = layers.reshape(
                x=edge_bias,
                shape=[0, -1, q.shape[1], q.shape[3]],
                inplace=True)
            # 5. transpose edge_bias from [M, B, N, H] to [B, N, M, H]
            edge_bias = layers.transpose(x=edge_bias, perm=[1, 2, 0, 3])
            out += edge_bias

        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = edge_aware_self_attention(q, k, v, edges_key, edges_value,
                                               attn_bias, d_key, dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Layer.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0',
                        initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           name=''):
    """
    Add residual connection, layer normalization and dropout to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head (self) attention and position-wise
    feed-forward sub-layers.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)))
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input,
                  edges_key,
                  edges_value,
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
                  preprocess_cmd="",
                  postprocess_cmd="dan",
                  param_initializer=None,
                  name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consists of a multi-head (self) attention sub-layer followed by
    a position-wise feed-forward sub-layer. Both two components are accompanied
    with the post_process_layer to add residual connection, layer normalization
    and dropout.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att'),
        None,
        None,
        edges_key,
        edges_value,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att')
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')
    return post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn')


def encoder(enc_input,
            edges_key,
            edges_value,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=None,
            name=''):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            edges_key,
            edges_value,
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
            name=name + '_layer_' + str(i))
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder")

    return enc_output
