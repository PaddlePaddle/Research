"""
    NACL patch for paddlenlp.transformers.llama.modeling.LlamaAttention.forward
    Pull request: https://github.com/PaddlePaddle/PaddleNLP/pull/8839
    Note: This patch can be removed when the above PR is merged.
"""
import math
import os
import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddlenlp
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
import scipy
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddlenlp.transformers.llama.modeling import apply_rotary_pos_emb, repeat_kv, scaled_dot_product_attention
from paddlenlp.transformers.llama import fusion_ops


def forward(
    self,
    hidden_states,
    position_ids: Optional[Tuple[paddle.Tensor]] = None,
    past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    alibi: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    npu_is_casual: bool = False,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

    if self.fuse_attention_qkv:
        mix_layer = self.qkv_proj(hidden_states)
        # NOTE for GQA attention fusion (compatible with MHA and MQA):
        # The weight for qkv_proj is in shape like [hidden_size, hidden_size + 2 * num_kv_heads * head_dim].
        # After the projection, the mix_layer is in shape like [b, s, hidden_size + 2 * num_kv_heads * head_dim].
        # Reshape the mix_layer into a shape like [b, s, num_kv_heads, (num_groups + 2) * head_dim],
        # where num_groups = num_q_heads // num_kv_heads.
        # Split the mix_layer on the last axis into three sections [num_groups * head_dim, head_dim, head_dim]
        # to represent the q, k and v respectively.
        # The q is in the shape like [b, s, num_kv_heads, num_groups * head_dim].
        # The k and v are in the shape like [b, s, num_kv_heads, head_dim].
        # Under MHA, the q is ready for the following calculation since num_kv_heads == num_q_heads,
        # But for the GQA or MQA, q should be reshaped into [b, s, num_q_heads, head_dim].
        if self.reshard_layer is not None:
            if self.sequence_parallel:
                assert self.seq_length % self.config.sep_parallel_degree == 0
                mix_layer = paddle.reshape_(
                    mix_layer,
                    [
                        -1,
                        self.seq_length // self.config.sep_parallel_degree,
                        self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
                    ],
                )
            # [bs, seq_len / sep, num_head, head_dim] -> [bs, seq_len, num_head / sep, head_dim]
            mix_layer = self.reshard_layer(
                mix_layer,
                split_axis=2,
                concat_axis=1,
            )
            mix_layer = paddle.reshape_(
                mix_layer, [0, self.seq_length, -1, (self.num_key_value_groups + 2) * self.head_dim]
            )  # [bs, seq_len, num_head/k, 3*head_dim], k is sep degree
        else:
            if self.sequence_parallel:
                target_shape = [
                    -1,
                    self.seq_length,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            else:
                target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
            mix_layer = paddle.reshape_(mix_layer, target_shape)
        query_states, key_states, value_states = paddle.split(
            mix_layer,
            num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
            axis=-1,
        )
        if self.gqa_or_mqa:
            query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.reshard_layer is not None:
            if self.sequence_parallel:
                assert self.seq_length % self.config.sep_parallel_degree == 0
                query_states = paddle.reshape(
                    query_states,
                    [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                )
                key_states = paddle.reshape(
                    key_states,
                    [
                        -1,
                        self.seq_length // self.config.sep_parallel_degree,
                        self.num_key_value_heads * self.head_dim,
                    ],
                )
                value_states = paddle.reshape(
                    value_states,
                    [
                        -1,
                        self.seq_length // self.config.sep_parallel_degree,
                        self.num_key_value_heads * self.head_dim,
                    ],
                )
            query_states = self.reshard_layer(
                query_states,
                split_axis=2,
                concat_axis=1,
            )
            key_states = self.reshard_layer(
                key_states,
                split_axis=2,
                concat_axis=1,
            )
            value_states = self.reshard_layer(
                value_states,
                split_axis=2,
                concat_axis=1,
            )
            query_states = paddle.reshape(
                query_states, [0, self.seq_length, -1, self.head_dim]
            )  # [bs, seq_len, num_head/k, head_dim], k is sep degree
            key_states = paddle.reshape(key_states, [0, self.seq_length, -1, self.head_dim])
            value_states = paddle.reshape(value_states, [0, self.seq_length, -1, self.head_dim])
        else:
            if self.sequence_parallel:
                target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
                target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
            else:
                target_query_shape = [0, 0, self.num_heads, self.head_dim]
                target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = query_states.reshape(shape=target_query_shape)
            key_states = key_states.reshape(shape=target_key_value_shape)
            value_states = value_states.reshape(shape=target_key_value_shape)

    kv_seq_len = key_states.shape[-3]

    if past_key_value is not None:
        if hasattr(self.config, "kvcache_eviction"):
            kv_seq_len += position_ids[0, 0]
        else:
            kv_seq_len += past_key_value[0].shape[-3]

    if self.config.rope:
        if self.reshard_layer is not None:
            batch_size, seq_length, _, _ = query_states.shape
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
        if self.config.context_parallel_degree > 1:
            batch_size, seq_length, _, _ = query_states.shape
            group = fleet.get_hybrid_communicate_group().get_sep_parallel_group()
            chunk_size = seq_length // 2
            chunk_num = group.nranks * 2
            rank = group.rank
            first_chunk_ids = paddle.arange(rank * chunk_size, (rank + 1) * chunk_size, dtype="int64")
            second_chunk_ids = paddle.arange(
                (chunk_num - rank - 1) * chunk_size, (chunk_num - rank) * chunk_size, dtype="int64"
            )
            position_ids = paddle.concat([first_chunk_ids, second_chunk_ids]).expand((batch_size, seq_length))
        if self.use_fused_rope:
            query_states, key_states = fusion_ops.fusion_rope(
                query_states,
                key_states,
                value_states,
                hidden_states,
                position_ids,
                past_key_value,
                self.rotary_emb,
                self.config.context_parallel_degree,
            )

        else:
            if self.config.context_parallel_degree > 1:
                kv_seq_len *= self.config.context_parallel_degree
            if self.config.use_long_sequence_strategies:
                cos, sin = self.rotary_emb(seq_len=kv_seq_len)
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]
                cos, sin = (
                    cos.cast(value_states.dtype) if cos.dtype != value_states.dtype else cos,
                    sin.cast(value_states.dtype) if sin.dtype != value_states.dtype else sin,
                )
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # [bs, seq_len, num_head, head_dim]
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        if self.config.immediate_clear_past_key_value:
            past_key_value[0]._clear_data()
            past_key_value[1]._clear_data()

    if self.kv_indices is not None:
        key_states = paddle.index_select(key_states, self.kv_indices, axis=2)
        value_states = paddle.index_select(value_states, self.kv_indices, axis=2)

    # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
    # repeat k/v heads if n_kv_heads < n_heads
    # paddle version > 2.6 or develop support flash-attn with gqa/mqa
    paddle_version = float(paddle.__version__[:3])
    if not self.config.use_flash_attention or ((paddle_version != 0.0) and (paddle_version <= 2.6)):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
    if (
        self.enable_recompute
        and self.layerwise_recompute
        and has_gradient
        and self.recompute_granularity == "core_attn"
    ):
        outputs = recompute(
            scaled_dot_product_attention,
            query_states,
            self.config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            alibi,
            attn_mask_startend_row_indices,
            self.sequence_parallel,
            reshard_layer=self.reshard_layer,
            use_reentrant=self.config.recompute_use_reentrant,
        )
    else:
        outputs = scaled_dot_product_attention(
            query_states,
            self.config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            alibi,
            attn_mask_startend_row_indices,
            self.sequence_parallel,
            reshard_layer=self.reshard_layer,
            npu_is_casual=npu_is_casual,
        )

    if (
        past_key_value is None
        and hasattr(self.config, "kvcache_eviction")
        and key_states.shape[1] > self.config.kvcache_eviction["min_eviction_seqlen"]
    ):
        key_states, value_states = self.kvcache_eviction(query_states, key_states, value_states)

    past_key_value = (key_states, value_states) if use_cache else None

    if output_attentions:
        attn_output, attn_weights = outputs
    else:
        attn_output = outputs

    # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
    # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    outputs = (attn_output,)

    if output_attentions:
        outputs += (attn_weights,)

    if use_cache:
        outputs += (past_key_value,)

    if type(outputs) is tuple and len(outputs) == 1:
        outputs = outputs[0]

    return outputs

def kvcache_eviction(self, query_states, key_states, value_states):
    """
    ACL2024
    NACL: A General and Effective KV Cache Eviction Framework for LLM sat Inference Time
    """

    q_len = key_states.shape[1]

    proxy_tokens_ratio = self.config.kvcache_eviction["proxy_tokens_ratio"]
    sink_tokens = self.config.kvcache_eviction["sink_tokens"]
    proxy_token_keep_ratio = self.config.kvcache_eviction["proxy_token_keep_ratio"]
    random_token_keep_ratio = self.config.kvcache_eviction["random_token_keep_ratio"]
    token_protect_ratio = self.config.kvcache_eviction["token_protect_ratio"]

    proxy_tokens = int(proxy_tokens_ratio * q_len)
    recent_protect_tokens = int(token_protect_ratio * q_len) - sink_tokens
    proxy_eviction_keep_tokens = int(proxy_token_keep_ratio * q_len)
    random_eviction_keep_tokens = int(random_token_keep_ratio * q_len)

    kvcache_buffer = sink_tokens + recent_protect_tokens + proxy_eviction_keep_tokens + random_eviction_keep_tokens

    evict_tokens = key_states.shape[1] - kvcache_buffer
    assert evict_tokens > 0, "number of evict_tokens must greater than 0"

    proxy_start_pos = key_states.shape[1] - recent_protect_tokens

    proxy_query_states = query_states[:, -proxy_tokens:, :]
    (_, _, softmax_lse, _) = paddle._C_ops.flash_attn(
        proxy_query_states,
        key_states,
        value_states,
        (None,),  # fixed_seed_offset
        None,  # attn_mask
        0.0,  # dropout
        False,  # causal
        False,  # return_softmax
        False,  # is_test
        "",
    )

    proxy_score = paddle.nn.functional.flash_attention.calc_reduced_attention_scores(
        proxy_query_states, key_states, softmax_lse
    )

    sink_keep_idx = np.arange(sink_tokens)
    recent_keep_idx = np.arange(proxy_start_pos, proxy_start_pos + recent_protect_tokens)

    index = []
    for head_idx in range(key_states.shape[2]):
        proxy_score_cur_head = proxy_score[:, head_idx].squeeze()
        proxy_score_cur_head = proxy_score_cur_head[sink_tokens:-recent_protect_tokens]
        topk_score, topk_idx = paddle.topk(proxy_score_cur_head, k=proxy_eviction_keep_tokens)
        proxy_eviction_keep_idx = topk_idx.numpy()

        to_evict_tokens_num = proxy_score_cur_head.shape[-1]
        idx_item_proxy_removed = np.delete(list(range(to_evict_tokens_num)), proxy_eviction_keep_idx)
        reserved_score = np.delete(proxy_score_cur_head.numpy(), proxy_eviction_keep_idx)
        reserved_score = scipy.special.softmax(reserved_score)
        random_eviction_keep_idx = np.random.choice(
            idx_item_proxy_removed, size=random_eviction_keep_tokens, replace=False, p=reserved_score
        )
        proxy_eviction_keep_idx = proxy_eviction_keep_idx + sink_tokens
        random_eviction_keep_idx = random_eviction_keep_idx + sink_tokens

        index_item = np.concatenate(
            [
                sink_keep_idx,
                np.sort(np.concatenate([proxy_eviction_keep_idx, random_eviction_keep_idx], axis=-1)),
                recent_keep_idx,
            ],
            axis=-1,
        )
        index.append(index_item)

    index = paddle.to_tensor(index).reshape([1, len(index), kvcache_buffer, 1])

    key_states = paddle.take_along_axis(key_states.transpose((0, 2, 1, 3)), index, axis=2).transpose((0, 2, 1, 3))
    value_states = paddle.take_along_axis(value_states.transpose((0, 2, 1, 3)), index, axis=2).transpose(
        (0, 2, 1, 3)
    )

    return key_states, value_states

def replace_llama_attn_with_nacl_eviction():
    paddlenlp.transformers.llama.modeling.LlamaAttention.kvcache_eviction = kvcache_eviction
    paddlenlp.transformers.llama.modeling.LlamaAttention.forward = forward
