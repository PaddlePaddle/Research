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
"""Multi-head Attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers as layers


def wrap_layer_with_block(layer, block_idx):
    """
    Make layer define support indicating block, by which we can add layers
    to other blocks within current block. This will make it easy to define
    cache among while loop.
    """

    class BlockGuard(object):
        """
        BlockGuard class.

        BlockGuard class is used to switch to the given block in a program by
        using the Python `with` keyword.
        """

        def __init__(self, block_idx=None, main_program=None):
            self.main_program = fluid.default_main_program(
            ) if main_program is None else main_program
            self.old_block_idx = self.main_program.current_block().idx
            self.new_block_idx = block_idx

        def __enter__(self):
            self.main_program.current_block_idx = self.new_block_idx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.main_program.current_block_idx = self.old_block_idx
            if exc_type is not None:
                return False  # re-raise exception
            return True

    def layer_wrapper(*args, **kwargs):
        """layer wrapper"""
        with BlockGuard(block_idx):
            return layer(*args, **kwargs)

    return layer_wrapper


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
                         static_kv=False,
                         param_initializer=None,
                         name='multi_head_att'):
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
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=fluid.ParamAttr(
                          name=name + '_query_fc.b_0'))

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.fc

        k = fc_layer(input=keys,
                     size=d_key * n_head,
                     num_flatten_dims=2,
                     param_attr=fluid.ParamAttr(
                         name=name + '_key_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=fluid.ParamAttr(
                         name=name + '_key_fc.b_0'))

        v = fc_layer(input=values,
                     size=d_value * n_head,
                     num_flatten_dims=2,
                     param_attr=fluid.ParamAttr(
                         name=name + '_value_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=fluid.ParamAttr(
                         name=name + '_value_fc.b_0'))

        return q, k, v

    def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.reshape

        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
                parent_idx) if cache is not None and static_kv else layers.transpose

        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])

        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:  # only for faster inference
            if static_kv:  # For encoder-decoder attention in inference
                cache_k, cache_v = cache["static_k"], cache["static_v"]
                # To init the static_k and static_v in cache.
                # Maybe we can use condition_op(if_else) to do these at the first
                # step in while loop to replace these, however it might be less
                # efficient.
                static_cache_init = wrap_layer_with_block(
                    layers.assign,
                    fluid.default_main_program().current_block().parent_idx)
                static_cache_init(k, cache_k)
                static_cache_init(v, cache_v)
            else:  # For decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]

            # gather cell states corresponding to selected parent
            select_k = layers.gather(cache_k, index=gather_idx)
            select_v = layers.gather(cache_v, index=gather_idx)
            if not static_kv:
                # For self attention in inference, use cache and concat time steps.
                select_k = layers.concat([select_k, k], axis=2)
                select_v = layers.concat([select_v, v], axis=2)

            # update cell states(caches) cached in global block
            layers.assign(select_k, cache_k)
            layers.assign(select_v, cache_v)
            return q, select_k, select_v
        return q, k, v

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key ** -0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0'),
                         bias_attr=fluid.ParamAttr(
                             name=name + '_output_fc.b_0')
                         )
    return proj_out


def multi_head_pooling(keys,
                       values,
                       attn_bias,
                       d_value,
                       d_model,
                       n_head=1,
                       dropout_rate=0.,
                       param_initializer=None,
                       name='multi_head_pooling'):
    """
    Multi-Head pooling. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    keys: # (batch_size, key_len, emb_dim)
    values: # (batch_size, key_len, emb_dim)
    attn_bias: # (batch_size, n_head, key_len, key_len)
    """
    values = keys if values is None else values

    if not (len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_kv(keys, values, n_head, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        k = layers.fc(input=keys,
                      size=n_head,
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
        return k, v

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
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def dot_product_pooling(k, v, attn_bias, dropout_rate):
        """
        Scaled Dot-Product Attention
        :param k:  (batch_size, n_head, key_len, 1)
        :param v:  (batch_size, n_head, key_len, dim_per_head)
        :param attn_bias:  (batch_size, n_head, key_len, key_len)
        :param dropout_rate:
        :param is_test:
        :return:
        """
        product = layers.squeeze(k, axes=[3])  # (batch_size, n_head, key_len)
        if attn_bias:
            # (batch_size, n_head, 1, key_len)
            attn_bias_sliced = fluid.layers.slice(attn_bias, axes=[2], starts=[0], ends=[1])
            product += layers.squeeze(attn_bias_sliced, axes=[2])  # (batch_size, n_head, key_len)

        weights = layers.softmax(product)  # (batch_size, n_head, key_len)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)

        pooling_out = layers.elementwise_mul(x=v, y=weights, axis=0)  # (batch_size, n_head, key_len, dim_per_head)
        pooling_out = layers.reduce_sum(pooling_out, dim=[2])  # (batch_size, n_head, dim_per_head)

        return pooling_out

    k, v = __compute_kv(keys, values, n_head, d_value)
    k = __split_heads(k, n_head)  # (batch_size, n_head, key_len, 1)
    v = __split_heads(v, n_head)  # (batch_size, n_head, key_len, dim_per_head)

    ctx_multiheads = dot_product_pooling(k, v, attn_bias, dropout_rate)  # (batch_size, n_head, dim_per_head)
    out = layers.reshape(ctx_multiheads, shape=[0, d_model])  # (batch_size, emb_dim)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=1,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out  # (batch_size*n_blocks, d_model)


def multi_head_structure_attention(queries,
                                   keys,
                                   values,
                                   attn_bias,
                                   graph_attn_bias,
                                   pos_win,
                                   d_key,
                                   d_value,
                                   d_model,
                                   n_head,
                                   dropout_rate=0.,
                                   param_initializer=None,
                                   name='multi_head_struct_att'):
    """
    Multi-Head Graph Attention module, regularized by graph structure
    Also includes several additional tricks.
    :param queries:  (batch_size, n_blocks, emb_dim)
    :param attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    :param graph_attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    :return:
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
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def graph_scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate,
                                           graph_attn_bias, pos_win):
        """
        Graph Scaled Dot-Product Attention
        :param q:  (batch_size, n_head, n_block, dim_per_head)
        :param k:  (batch_size, n_head, n_block, dim_per_head)
        :param v:  (batch_size, n_head, n_block, dim_per_head)
        :param attn_bias:  (batch_size, n_head, n_block, n_block)
        :param d_key:
        :param dropout_rate:
        :param graph_attn_bias:  (batch_size, n_head, n_block, n_block)
        :param pos_win:
        :return:
        """
        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)  # (batch_size, n_head, n_block, dim_per_head)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias  # (batch_size, n_head, n_block, n_block)

        if graph_attn_bias:
            # re-weight the attention score with gaussian weights
            gaussian_w = (-0.5 * graph_attn_bias * graph_attn_bias) / (
                    (0.5 * pos_win) ** 2)  # [batch, n_heads, n_block, n_block]
            product += gaussian_w

        weights = layers.softmax(product)  # [batch, n_heads, n_block, n_block]

        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)

        graph_out = layers.matmul(weights, v)  # [batch, n_heads, n_block, dim_per_head]
        return graph_out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    q = __split_heads(q, n_head)  # (batch_size, n_head, n_block, dim_per_head)
    k = __split_heads(k, n_head)  # (batch_size, n_head, n_block, dim_per_head)
    v = __split_heads(v, n_head)  # (batch_size, n_head, n_block, dim_per_head)

    ctx_multiheads = graph_scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                        dropout_rate, graph_attn_bias, pos_win)

    out = __combine_heads(ctx_multiheads)  # (batch_size, n_block, d_model)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out  # (batch_size, n_block, d_model)


def multi_head_hierarchical_attention(queries,
                                      keys_w,
                                      values_w,
                                      attn_bias_w,
                                      keys_s,
                                      values_s,
                                      attn_bias_s,
                                      graph_attn_bias,
                                      pos_win,
                                      d_key,
                                      d_value,
                                      d_model,
                                      n_head=1,
                                      dropout_rate=0.,
                                      dropout_seed=None,
                                      cache=None,
                                      gather_idx=None,
                                      param_initializer=None,
                                      name='multi_head_hier_att'):
    """
    Multi-Head Hierarchical Attention module, consists of word-level and sentence-level
    word-level attention is multi-head attention, normalized by sentence-level
    sentence-level attention is multi-head attention, regularized by graph structure.
    :param queries:  (batch_size, tgt_len, emb_dim)
    :param keys_w:  (batch_size, n_blocks, n_tokens, emb_dim)
    :param values_w:  (batch_size, n_blocks, n_tokens, emb_dim)
    :param attn_bias_w:  (batch_size, n_blocks, n_head, tgt_len, n_tokens)
    :param keys_s:  (batch_size, n_blocks, emb_dim)
    :param values_s:  (batch_size, n_blocks, emb_dim)
    :param attn_bias_s:  (batch_size, n_head, tgt_len, n_blocks)
    :param graph_attn_bias: (batch_size, n_head, n_blocks, n_blocks)
    :return:
    """
    if not (len(keys_w.shape) == len(values_w.shape) == 4):
        raise ValueError(
            "Inputs: keys_w and values_w should all be 4-D tensors.")

    if not (len(queries.shape) == len(keys_s.shape) == len(values_s.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys_s and values_s should all be 3-D tensors.")

    batch_size = layers.shape(attn_bias_w)[0]
    key_s_len = layers.shape(keys_s)[1]
    query_len = layers.shape(queries)[1]
    assert d_model // n_head == d_key and d_model % n_head == 0, "the head count is invalid"

    def __compute_qkv_sent(queries, keys, values, n_head, d_key, d_value, suffix='_sen'):
        """
        Add linear projection to queries, keys, and values.
        :param queries: (batch_size, tgt_len, d_model)
        :param keys:  (batch_size, n_block, d_model)
        :param values: (batch_size, n_block, d_model)
        :return:
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + suffix + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + suffix + '_query_fc.b_0')

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None else layers.fc

        k = fc_layer(input=keys,
                     size=d_key * n_head,
                     num_flatten_dims=2,
                     param_attr=fluid.ParamAttr(
                         name=name + suffix + '_key_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=name + suffix + '_key_fc.b_0')

        v = fc_layer(input=values,
                     size=d_value * n_head,
                     num_flatten_dims=2,
                     param_attr=fluid.ParamAttr(
                         name=name + suffix + '_value_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=name + suffix + '_value_fc.b_0')
        return q, k, v

    def __split_heads_qkv_sent(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose.
        :param queries: (batch_size, tgt_len, d_model)
        :param keys:  (batch_size, n_block, d_model)
        :param values: (batch_size, n_block, d_model)
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None else layers.reshape

        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
                parent_idx) if cache is not None else layers.transpose

        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])

        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:  # only for faster inference
            cache_k, cache_v = cache["static_k_sent"], cache["static_v_sent"]
            # To init the static_k and static_v in cache.
            static_cache_init = wrap_layer_with_block(
                layers.assign,
                fluid.default_main_program().current_block().parent_idx)
            static_cache_init(k, cache_k)
            static_cache_init(v, cache_v)

            # gather cell states corresponding to selected parent
            select_k = layers.gather(cache_k, index=gather_idx)
            select_v = layers.gather(cache_v, index=gather_idx)

            layers.assign(select_k, cache_k)
            layers.assign(select_v, cache_v)
            return q, select_k, select_v
        return q, k, v

    def __compute_graph_bias(q, graph_attn_mask, pos_win):
        """
        :param q: (batch_size, n_heads, query_len, dim_per_head)
        :param graph_attn_mask: (batch_size, n_head, key_s_len, key_s_len)
        :param pos_win:
        :return:
        """
        # (batch_size, n_heads, query_len, dim_per_head)
        pos_v = layers.fc(input=q,
                          size=d_value,
                          num_flatten_dims=3,
                          param_attr=fluid.ParamAttr(
                              name=name + '_pos_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=name + '_pos_fc.b_0')

        # (batch_size, n_heads, query_len, 1)
        pos_s = layers.fc(input=layers.tanh(pos_v),
                          size=1,
                          num_flatten_dims=3,
                          param_attr=fluid.ParamAttr(
                              name=name + '_pos_score_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=False)

        # (batch_size, n_heads, query_len, 1)
        pos = layers.sigmoid(pos_s) * (key_s_len - 1)

        # (batch_size, n_heads, query_len, 1)
        pos_up = layers.cast(layers.ceil(pos), dtype='int64')
        # print("pos_up.shape = %s" % str(pos_up.shape))
        pos_down = layers.cast(layers.floor(pos), dtype='int64')
        # print("pos_down.shape = %s" % str(pos_down.shape))

        batch_ind = layers.range(0, layers.cast(batch_size, dtype='int64'), 1, 'int64')
        # print("batch_ind.shape = %s" % str(batch_ind.shape))
        batch_ind = layers.unsqueeze(batch_ind, axes=[1, 2, 3])  # (batch_size, 1, 1, 1)
        batch_ind = layers.expand(batch_ind,
                                  expand_times=[1, n_head, query_len, 1])  # (batch_size, n_heads, query_len, 1)
        # print("batch_ind.shape = %s" % str(batch_ind.shape))

        head_ind = layers.range(0, n_head, 1, 'int64')
        # print("head_ind.shape = %s" % str(head_ind.shape))
        head_ind = layers.unsqueeze(head_ind, axes=[0, 2, 3])  # (1, n_heads, 1, 1)
        head_ind = layers.expand(head_ind, expand_times=[batch_size, 1, query_len, 1])
        # print("head_ind.shape = %s" % str(head_ind.shape))

        query_ind = layers.range(0, layers.cast(query_len, dtype='int64'), 1, 'int64')
        # print("query_ind.shape = %s" % str(query_ind.shape))
        query_ind = layers.unsqueeze(query_ind, axes=[0, 1, 3])  # (1, 1, query_len, 1)
        query_ind = layers.expand(query_ind, expand_times=[batch_size, n_head, 1, 1])
        # print("query_ind.shape = %s" % str(query_ind.shape))

        # (batch_size, n_heads, query_len, 4)
        pos_up_ind = layers.concat(input=[batch_ind, head_ind, query_ind, pos_up], axis=-1)
        # print("pos_up_ind.shape = %s" % str(pos_up_ind.shape))
        pos_up_ind.stop_gradient = True
        pos_down_ind = layers.concat(input=[batch_ind, head_ind, query_ind, pos_down], axis=-1)
        # print("pos_down_ind.shape = %s" % str(pos_down_ind.shape))
        pos_down_ind.stop_gradient = True

        # (batch_size, n_heads, query_len, key_s_len, key_s_len)
        graph_attn_mask = layers.unsqueeze(graph_attn_mask, axes=[2])
        # print("graph_attn_mask.shape = %s" % str(graph_attn_mask.shape))
        graph_attn_mask = layers.expand(graph_attn_mask, expand_times=[1, 1, query_len, 1, 1])
        # print("graph_attn_mask.shape = %s" % str(graph_attn_mask.shape))

        # (batch_size, n_heads, query_len, key_s_len)
        graph_attn_mask_up = layers.gather_nd(input=graph_attn_mask, index=pos_up_ind)
        graph_attn_mask_down = layers.gather_nd(input=graph_attn_mask, index=pos_down_ind)

        # print("graph_attn_mask_up.shape = %s" % str(graph_attn_mask_up.shape))
        # print("graph_attn_mask_down.shape = %s" % str(graph_attn_mask_down.shape))
        # print("pos_up.shape = %s" % str(pos_up.shape))
        # print("pos_down.shape = %s" % str(pos_down.shape))

        # linearly combine up and down (batch_size, n_heads, query_len, key_s_len)
        graph_attn_mask_select = graph_attn_mask_up * (1.0 - (layers.cast(pos_up, dtype='float32') - pos)) + \
                                 graph_attn_mask_down * (1.0 - (pos - layers.cast(pos_down, dtype='float32')))
        # print("graph_attn_mask_select.shape = %s" % str(graph_attn_mask_select.shape))
        # re-weight the attention score with gaussian weights
        gaussian_w = (-0.5 * graph_attn_mask_select * graph_attn_mask_select) / (
                (0.5 * pos_win) ** 2)  # [batch, n_heads, query_len, key_s_len]
        # print("gaussian_w.shape = %s" % str(gaussian_w.shape))

        return gaussian_w

    def __combine_heads_sent(x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def graph_scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate,
                                           graph_attn_bias, pos_win):
        """
        Graph Scaled Dot-Product Attention
        :param q: (batch_size, n_heads, query_len, dim_per_head)
        :param k: (batch_size, n_heads, key_s_len, dim_per_head)
        :param v: (batch_size, n_heads, key_s_len, dim_per_head)
        :param attn_bias: (batch_size, n_head, query_len, key_s_len)
        :param graph_attn_bias: (batch_size, n_head, key_s_len, key_s_len)
        :return:
            proj_out: [batch, n_heads, query_len, dim_per_hed]
            weights: [batch, n_heads, query_len, key_s_len]
        """
        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)  # (batch_size, n_heads, tgt_len, n_block)
        if attn_bias:
            product += attn_bias  # (batch_size, n_heads, tgt_len, n_block)

        if graph_attn_bias:
            # re-weight the attention score with gaussian weights
            gaussian_w = __compute_graph_bias(scaled_q, graph_attn_bias, pos_win)
            product += gaussian_w  # [batch, n_heads, query_len, key_s_len]

        weights = layers.softmax(product)  # [batch, n_heads, query_len, key_s_len]

        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                dropout_implementation="upscale_in_train",
                is_test=False)

        out = layers.matmul(weights, v)  # [batch, n_heads, query_len, dim_per_hed]

        # Project back to the model size.
        combine_out = __combine_heads_sent(out)  # (batch_size, query_len, emb_dim)
        proj_out = layers.fc(input=combine_out,
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_sen_fc.w_0',
                                 initializer=param_initializer),
                             bias_attr=name + '_sen_fc.b_0')

        return proj_out, weights

    # sentence-level graph attention
    # (batch_size, query_len, d_model)
    q_s, k_s, v_s = __compute_qkv_sent(queries, keys_s, values_s,
                                       n_head, d_key, d_value, suffix='_sen')

    # (batch_size, n_heads, query_len, dim_per_head)
    q_s, k_s, v_s = __split_heads_qkv_sent(q_s, k_s, v_s, n_head, d_key, d_value)

    context_s, attn_s = graph_scaled_dot_product_attention(q_s, k_s, v_s, attn_bias_s, d_key,
                                                           dropout_rate, graph_attn_bias, pos_win)

    def __compute_qkv_word(queries, keys, values, n_head, d_key, d_value, suffix='_word'):
        """
        Add linear projection to queries, keys, and values.
        :param queries: (batch_size, tgt_len, d_model)
        :param keys:  (batch_size, n_block, n_token, d_model)
        :param values: (batch_size, n_block, n_token, d_model)
        :return:
        """

        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + suffix + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + suffix + '_query_fc.b_0')

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None else layers.fc

        k = fc_layer(input=keys,
                     size=d_key * n_head,
                     num_flatten_dims=3,
                     param_attr=fluid.ParamAttr(
                         name=name + suffix + '_key_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=name + suffix + '_key_fc.b_0')

        v = fc_layer(input=values,
                     size=d_value * n_head,
                     num_flatten_dims=3,
                     param_attr=fluid.ParamAttr(
                         name=name + suffix + '_value_fc.w_0',
                         initializer=param_initializer),
                     bias_attr=name + suffix + '_value_fc.b_0')
        return q, k, v

    def __split_heads_qkv_word(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose.
        :param queries: (batch_size, tgt_len, d_model)
        :param keys:  (batch_size, n_block, n_token, d_model)
        :param values: (batch_size, n_block, n_token, d_model)
        :return:
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # [batch_size, n_head, tgt_len, dim_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])

        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None else layers.reshape

        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
                parent_idx) if cache is not None else layers.transpose

        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 1, 3, 2, 4])

        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 1, 3, 2, 4])

        if cache is not None:  # only for faster inference
            cache_k, cache_v = cache["static_k_word"], cache["static_v_word"]
            # To init the static_k and static_v in cache.
            static_cache_init = wrap_layer_with_block(
                layers.assign,
                fluid.default_main_program().current_block().parent_idx)
            static_cache_init(k, cache_k)
            static_cache_init(v, cache_v)

            # gather cell states corresponding to selected parent
            select_k = layers.gather(cache_k, index=gather_idx)
            select_v = layers.gather(cache_v, index=gather_idx)

            layers.assign(select_k, cache_k)
            layers.assign(select_v, cache_v)
            return q, select_k, select_v
        return q, k, v

    def __combine_heads_word(x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention_with_sen_norm(q, k, v, attn_bias, d_key, dropout_rate, attn_s):
        """
        Scaled Dot-Product Attention with sentence-level normalize
        :param q: (batch_size, n_head, tgt_len, dim_per_head)
        :param k: (batch_size, n_blocks, n_head, n_tokens, dim_per_head)
        :param v: (batch_size, n_blocks, n_head, n_tokens, dim_per_head)
        :param attn_bias: (batch_size, n_blocks, n_head, tgt_len, n_tokens)
        :param attn_s:  [batch, n_heads, query_len, key_s_len]
        :return:
        """
        # print("q.shape = %s" % str(q.shape))
        # (batch_size, n_block, n_head, tgt_len, dim_per_head)
        q = layers.expand(layers.unsqueeze(q, axes=[1]), expand_times=[1, key_s_len, 1, 1, 1])
        # print("q.shape = %s" % str(q.shape))
        # (batch_size*n_block, n_head, tgt_len, dim_per_head)
        # q = layers.reshape(q, shape=[-1, n_head, query_len, d_key])
        # print("q.shape = %s" % str(q.shape))

        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)

        # (batch_size, n_block, n_head, tgt_len, n_token)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)

        if attn_bias:
            product += attn_bias  # (batch_size, n_block, n_head, tgt_len, n_token)

        weights = layers.softmax(product)  # (batch_size, n_block, n_head, tgt_len, n_token)

        # attn_w = layers.reshape(weights, shape=[batch_size, key_s_len, n_head, query_len, -1])
        # (batch_size, n_head, tgt_len, n_block, n_token)
        attn_w = layers.transpose(weights, perm=[0, 2, 3, 1, 4])
        # (batch_size, n_head, tgt_len, n_block, n_token)
        attn_w = layers.elementwise_mul(attn_w, layers.unsqueeze(attn_s, axes=[-1]), axis=0)
        # (batch_size, n_head, tgt_len, n_block*n_token)
        attn_w = layers.reshape(attn_w, shape=[batch_size, n_head, query_len, -1])

        if dropout_rate:
            attn_w = layers.dropout(  # (batch_size, n_head, tgt_len, n_block*n_token)
                attn_w,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                dropout_implementation="upscale_in_train",
                is_test=False)

        # values_w = layers.reshape(v, shape=[batch_size, key_s_len, n_head, -1, d_value])
        values_w = layers.transpose(v, perm=[0, 2, 1, 3, 4])
        # (batch_size, n_head, n_block*n_token, dim_per_head)
        values_w = layers.reshape(values_w, shape=[batch_size, n_head, -1, d_value])

        out = layers.matmul(attn_w, values_w)  # (batch_size, n_head, tgt_len, dim_per_head)

        # Project back to the model size.
        combine_out = __combine_heads_word(out)  # (batch_size, query_len, emb_dim)

        proj_out = layers.fc(input=combine_out,  # (batch_size, tgt_len, model_dim)
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_word_fc.w_0',
                                 initializer=param_initializer),
                             bias_attr=name + '_word_fc.b_0')

        return proj_out, attn_w

    # word-level normalized attention
    q_w, k_w, v_w = __compute_qkv_word(queries, keys_w, values_w,
                                       n_head, d_key, d_value, suffix='_word')
    # (batch_size, n_block, n_heads, n_token, dim_per_head)
    q_w, k_w, v_w = __split_heads_qkv_word(q_w, k_w, v_w, n_head, d_key, d_value)

    context_w, attn_w = scaled_dot_product_attention_with_sen_norm(q_w, k_w, v_w, attn_bias_w, d_key,
                                                                   dropout_rate, attn_s)

    combine_out = layers.concat([context_s, context_w], axis=2)  # (batch_size, query_len, 2*emb_dim)
    # Project back to the model size.
    final_out = layers.fc(input=combine_out,  # (batch_size, query_len, emb_dim)
                          size=d_model,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + '_cat_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=name + '_cat_fc.b_0')

    return final_out  # (batch_size, query_len, emb_dim)

