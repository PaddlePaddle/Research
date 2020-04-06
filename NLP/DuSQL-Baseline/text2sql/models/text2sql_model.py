#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""text2sql model based on seq2seq framework
"""

import sys
import os
import traceback
import logging
import functools
from collections import OrderedDict

import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.layers import tensor

from text2sql import g
from text2sql.framework.rule import InstanceName as C    # Constant names
from text2sql.framework.model import Model as BaseModel
from text2sql.framework.modules.ernie import ErnieModel, ErnieConfig

from text2sql import models
from text2sql.datalib import DName
from text2sql.grammar import Grammar
from text2sql.models import gmr_models
from text2sql.utils import fluider
from text2sql.utils import nn_utils

MAX_GRADIENT_NORM = 5.0

class Text2SQL(BaseModel):

    """Text2SQL main model"""

    def __init__(self, params_dict, save_predict_file=None):
        """init of class

        Args:
            params_dict (TYPE): NULL
            save_predict_file (str): 保存预测结果。仅在预测阶段使用。默认为 None


        """
        super(Text2SQL, self).__init__(params_dict)

        self.optimizer_params = params_dict['optimizer']
        self.hidden_size = params_dict["hidden_size"]
        self.grammar = Grammar(params_dict['grammar_file'])
        self._original_dropout = params_dict['dropout']
        self.dropout = self._original_dropout
        self.init_scale = -1  # '<0' means use paddle default initializer

        self.encoder_params = params_dict["encoder"]
        self.encoder_type = self.encoder_params["type"]
        if self.encoder_type == 'LSTM':
            self.embedding_size = self.encoder_params["embedding_size"]
            self.encoder_layers = self.encoder_params["encoder_layers"]
            self.table_enc_type = self.encoder_params.get("table_encoder", "simple_sum")
            self.table_attention = self.encoder_params["table_attention"]
            self.finetune_emb = self.encoder_params["finetune_emb"]

            with open(self.encoder_params['vocab_file']) as ifs:
                vocab = [x.rstrip('\n') for x in ifs]
            self.src_vocab_size = len(vocab)

            embedding_file = self.encoder_params.get('embedding_file', None)
            if embedding_file is None:      # uniform 初始化，Embedding 从头训
                self._emb_weight_init = nn_utils.uniform(self.init_scale)
            else:                           # 使用预训练的 Embedding 初始化
                dct_vocab = dict([(word, idx) for idx, word in enumerate(vocab)])
                weight_mat = np.random.normal(0., 1., size=[self.src_vocab_size, self.embedding_size])
                with open(embedding_file) as ifs:
                    for line in ifs:
                        lst_fields = line.rstrip().split(' ')
                        if len(lst_fields) != self.embedding_size + 1:
                            raise ValueError('embedding file format error: %s' % (line.rstrip('\n')))
                        word_id = dct_vocab[lst_fields[0]]
                        emb = [float(x) for x in lst_fields[1:]]
                        weight_mat[word_id, :] = emb
                self._emb_weight_init = fluid.initializer.NumpyArrayInitializer(weight_mat)
        elif self.encoder_type == "ErnieTokenEmbedding":
            self.table_enc_type = self.encoder_params.get("table_encoder", "simple_sum")
            self.table_attention = self.encoder_params["table_attention"]
        else:
            raise ValueError("Encoder Type Error. Expect LSTM/ErnieTokenEmbedding, bug got %s" % (self.encoder_type))

        self.decoder_params = params_dict['decoder']
        self.max_infer_step = self.decoder_params["max_infer_step"]
        self.lf_emb_size = self.decoder_params["lf_emb_size"]
        self.lf_name_emb_size = self.decoder_params["lf_name_emb_size"]
        self.beam_size = self.decoder_params["beam_size"]

        self._batch_size = None     # Variable, 会在 forward 中初始化
        self.all_inputs_name = None # list, 在 _rnn_encoder/_ernie_encoder 中初始化
        #### 运行时创建并初始化的成员 ########################
        ### self._read_question() --> self.question_xxx     ##
        ### self._read_table_name() --> self.tname_xxx      ##
        ### self._read_column_name() --> self.cname_xxx     ##
        ### self._read_value() --> self.value_xxx           ##

        self.ofs_predict_result = None
        if save_predict_file is not None:
            self.ofs_predict_result = open(save_predict_file, 'w')
        self.best_acc = -1
        self.best_step = -1
        self.best_epoch = -1

    def forward(self, slots_dict, phase):
        """model forward process

        Args:
            slots_dict (dict): NULL
            phase (str): NULL

        Returns: TODO

        Raises: NULL

        """
        if phase == 'training':
            self.dropout = self._original_dropout
        else:
            self.dropout = 0.0
        logging.info("%s stage dropout prob is %f", phase, self.dropout)

        self.question_fea = slots_dict["question_features"][C.RECORD_ID][C.SRC_IDS]
        self.table_fea = slots_dict["table_features"][C.RECORD_ID][C.SRC_IDS]
        self.column_fea = slots_dict["column_features"][C.RECORD_ID][C.SRC_IDS]
        self.value_fea = slots_dict["value_features"][C.RECORD_ID][C.SRC_IDS]
        self.column2table_mask = slots_dict["column_tables"][C.RECORD_ID][C.SRC_IDS]
        self.all_inputs_name = [self.column2table_mask.name]

        ## 获取 feature 的表示
        self.question_fea_emb, self.table_fea_emb, self.column_fea_emb, self.value_fea_emb = None, None, None, None
        if g.use_question_feature:
            logging.info("question feature is being used")
            self.question_fea_emb = self._feature_embedder(self.question_fea, name='question_fea')
            self.all_inputs_name.append(self.question_fea.name)
        if g.use_table_feature:
            logging.info("table feature is being used")
            self.table_fea_emb = self._feature_embedder(self.table_fea, name='table_fea')
            self.all_inputs_name.append(self.table_fea.name)
        if g.use_column_feature:
            logging.info("column feature is being used")
            self.column_fea_emb = self._feature_embedder(self.column_fea, name='column_fea')
            self.all_inputs_name.append(self.column_fea.name)
        if g.use_value_feature:
            logging.info("value feature is being used")
            self.value_fea_emb = self._feature_embedder(self.value_fea, name='value_fea')
            self.all_inputs_name.append(self.value_fea.name)

        if self.encoder_type == "LSTM":
            encoder_func = self._rnn_encoder
        elif self.encoder_type == "ErnieTokenEmbedding":
            encoder_func = self._ernie_encoder
        self.input_state, self.question_encoding, self.tname_encoding, self.cname_encoding, self.value_encoding = \
                                                                                           encoder_func(slots_dict)
        if phase != C.SAVE_INFERENCE:
            label_info = slots_dict["label"][C.RECORD_ID]
            self.train_label = label_info[DName.TRAIN_LABEL]
            self.infer_label = label_info[DName.INFER_LABEL]
            self.label_mask = label_info[DName.MASK_IDS]
            self.label_lens = label_info[DName.SEQ_LENS]
            self.infer_actions = label_info[DName.INFER_ACTIONS]
            self.infer_gmr_mask = label_info[DName.INFER_GMR_MASK]
            self.valid_table_mask = label_info[DName.INFER_COL2TABLE_MASK]

        ## decoder ##
        run_mode = 'infer' if phase == C.SAVE_INFERENCE else 'train'
        decode_output = self._decoder(self.question_encoding, self.input_state, run_mode, self.beam_size)

        ## 返回结果 ##
        forward_output_dict = {}
        if phase == C.SAVE_INFERENCE:
            forward_return_dict = {
                C.TARGET_FEED_NAMES: self.all_inputs_name,
                C.TARGET_PREDICTS: [decode_output]
            }
            logging.debug("target feed names: %s", self.all_inputs_name)
        else:    # train or test
            output_id = fluid.layers.argmax(decode_output, axis=2)
            self.loss = self._calc_loss(decode_output, self.infer_label, loss_type='softmax_ce')

            """PREDICT_RESULT,LABEL,LOSS 是关键字，必须要赋值并返回"""
            forward_return_dict = {
                C.PREDICT_RESULT: output_id,
                C.LABEL: self.infer_label,
                C.LOSS: self.loss
            }

        return forward_return_dict

    def optimizer(self, loss, is_fleet):
        """optimize by loss

        Args:
            loss (Tensor): NULL

        Returns: None

        Raises: NULL

        """
        opt_setting = OrderedDict()

        if self.encoder_type == 'LSTM':
            if MAX_GRADIENT_NORM > 0:
                fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=MAX_GRADIENT_NORM))
            optimizer = fluid.optimizer.Adam(learning_rate=self.optimizer_params["learning_rate"])
            optimizer.minimize(loss)
        else:   # ERNIE
            assert self.optimizer_params is not None, 'optimizer should be setting in config when use ernie encoder'
            opt_setting['use_ernie_opt'] = True

            ernie_opt_params = OrderedDict()
            ernie_opt_params["loss"] = loss
            ernie_opt_params["warmup_steps"] = self.optimizer_params["warmup_steps"]
            ernie_opt_params["num_train_steps"] = self.optimizer_params["max_train_steps"]
            ernie_opt_params["learning_rate"] = self.optimizer_params["learning_rate"]
            ernie_opt_params["weight_decay"] = self.optimizer_params["weight_decay"]
            ernie_opt_params["scheduler"] = self.optimizer_params["lr_scheduler"]
            ernie_opt_params["use_fp16"] = self.encoder_params.get("use_fp16", False)
            ernie_opt_params["use_dynamic_loss_scaling"] = self.optimizer_params["use_dynamic_loss_scaling"]
            ernie_opt_params["init_loss_scaling"] = self.optimizer_params["init_loss_scaling"]
            ernie_opt_params["incr_every_n_steps"] = self.optimizer_params["incr_every_n_steps"]
            ernie_opt_params["decr_every_n_nan_or_inf"] = self.optimizer_params["decr_every_n_nan_or_inf"]
            ernie_opt_params["incr_ratio"] = self.optimizer_params["incr_ratio"]
            ernie_opt_params["decr_ratio"] = self.optimizer_params["decr_ratio"]
            opt_setting['opt_args'] = ernie_opt_params

        return opt_setting

    def get_metrics(self, model_output, meta_info, phase):
        """calculate metrics, like acc

        Args:
            predictions (Tensor): NULL
            label (Tensor): NULL
            phase (string): NULL

        Returns: dict
            exact_match_acc --> LFAcc

        Raises: NULL

        """
        lst_prediction = model_output[C.PREDICT_RESULT]
        lst_label = model_output[C.LABEL]
        if phase == 'training':
            lst_prediction = [lst_prediction]
            lst_label = [lst_label]

        lst_loss = model_output[C.LOSS]
        loss = sum(lst_loss) / len(lst_loss)
        #print(">>> gold:", lst_label[0][0][:30].tolist())
        #print("  | pred:", lst_prediction[0][0][:30].tolist())
        sys.stdout.flush()
        cnt_correct = 0
        cnt_instances = 0
        for prediction, label in zip(lst_prediction, lst_label):
            elem_correct = prediction == label
            ins_correct = np.min(elem_correct, axis=1)
            cnt_correct += sum(ins_correct)
            cnt_instances += len(ins_correct)
        acc = cnt_correct / cnt_instances if cnt_instances != 0 else 0.0
        epoch = meta_info.get('epoch', 0)
        steps = meta_info['steps']
        cost_time = meta_info[C.TIME_COST]
        log_str_common = '[%.2fs]%s epoch %d steps %d: loss=%f, acc=%f' % (cost_time, phase, epoch, steps, loss, acc)
        if phase == 'training':
            logging.info(log_str_common)
        else:
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_epoch = epoch
                self.best_step = steps
            logging.info('%s. best=%f on epoch %d step %d',
                    log_str_common, self.best_acc, self.best_epoch, self.best_step)
        return OrderedDict(exact_match_acc=acc)

    def parse_predict_result(self, predict_result):
        """parse predict result

        Args:
            predict_result (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        if self.ofs_predict_result is None:
            raise ValueError("config <predictor.save_predict_file> is needed when perform inference")

        if type(predict_result) is not list:
            predict_result = [predict_result]

        for batch_res in predict_result:
            if type(batch_res) is not np.ndarray:
                np_result = np.array(predict_result[0].data.int64_data()).reshape(predict_result[0].shape)
            else:
                np_result = batch_res
            for ins in np_result:
                # 截取 END 之前的有效输出
                if self.grammar.END in ins:
                    end_idx = np.where(ins == self.grammar.END)[0][0]
                    ins = ins[: end_idx]
                while len(ins) > 0 and ins[-1] == 0:
                    ins = ins[: -1]

                ins_str = [str(x) for x in ins]
                self.ofs_predict_result.write(' '.join(ins_str) + '\n')

    def _rnn_encoder(self, slots_dict):
        """use rnn to encode question, tables/columns/values

        Args:
            slots_dict (dict): NULL

        Returns: TODO

        Raises: NULL
        """
        ## 获取原始输入 ##
        self._read_question(slots_dict["question_tokens"][C.RECORD_ID])
        self._read_table_name(slots_dict["table_names"][C.RECORD_ID])
        self._read_column_name(slots_dict["column_names"][C.RECORD_ID])
        self._read_value(slots_dict["values"][C.RECORD_ID])
        self.all_inputs_name += [self.question_ids.name, ## self.question_lens.name,
                                self.q_span_lens.name, self.q_span_pos.name,
                                self.q_span_tok_lens.name,
                                self.tname_ids.name, self.tname_item_lens.name,
                                self.tname_pos.name, self.tname_token_lens.name,
                                self.cname_ids.name, self.cname_item_lens.name,
                                self.cname_pos.name, self.cname_token_lens.name,
                                self.value_ids.name, self.value_item_lens.name,
                                self.value_pos.name, self.value_token_lens.name]
        if self.table_enc_type == 'birnn':
            self.all_inputs_name += [self.tname_lens.name, self.cname_lens.name, self.value_lens.name]

        ## 获取输入的 embedding ##
        self.question_emb = self._src_embedder(self.question_ids)
        self.tname_emb = self._src_embedder(self.tname_ids)
        self.cname_emb = self._src_embedder(self.cname_ids)
        self.value_emb = self._src_embedder(self.value_ids)

        ## question encoder ##
        question_encoding, question_enc_state = self._question_encoder()

        ## table/column/value encoder ##
        q_padding_mask = self.question_mask - 1.0
        tname_encoding, _ = self._table_encoder(self.tname_emb, self.tname_lens, self.tname_item_lens,
                self.tname_pos, self.tname_token_lens, self.table_fea_emb,
                name=('table_enc', 'table_enc_attn', 'tab_enc_out'),
                question_encoding=question_encoding, q_padding_mask=q_padding_mask)
        cname_encoding, _ = self._table_encoder(self.cname_emb, self.cname_lens, self.cname_item_lens,
                self.cname_pos, self.cname_token_lens, self.column_fea_emb,
                name=('table_enc', 'table_enc_attn', 'col_enc_out'),
                question_encoding=question_encoding, q_padding_mask=q_padding_mask)
        value_encoding, _ = self._table_encoder(self.value_emb, self.value_lens, self.value_item_lens,
                self.value_pos, self.value_token_lens, self.value_fea_emb,
                name=('table_enc', 'table_enc_attn', 'val_enc_out'),
                question_encoding=question_encoding, q_padding_mask=q_padding_mask)

        return [question_enc_state, question_encoding, tname_encoding, cname_encoding, value_encoding]

    def _ernie_encoder(self, slots_dict):
        """use ernie to encode question, tables/columns/values

        Args:
            slots_dict (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        batch_instance = slots_dict["question_tokens"][C.RECORD_ID]
        input_qtc_src = batch_instance[DName.QTC_IDS]
        input_qtc_pos = batch_instance[DName.QTC_POS_IDS]
        input_qtc_sent = batch_instance[DName.QTC_SENTENCE_IDS]
        input_qtc_mask = batch_instance[DName.QTC_MASK_IDS]
        input_qtc_task = batch_instance[DName.QTC_TASK_IDS]
        input_qv_src = batch_instance[DName.QV_IDS]
        input_qv_pos = batch_instance[DName.QV_POS_IDS]
        input_qv_sent = batch_instance[DName.QV_SENTENCE_IDS]
        input_qv_mask = batch_instance[DName.QV_MASK_IDS]
        input_qv_task = batch_instance[DName.QV_TASK_IDS]
        input_q_pos = batch_instance[DName.Q_POS]
        input_t_pos = batch_instance[DName.T_POS]
        input_c_pos = batch_instance[DName.C_POS]
        input_v_pos = batch_instance[DName.V_POS]
        q_span_lens = batch_instance[DName.Q_LEN]
        self.tname_item_lens = batch_instance[DName.T_LEN]
        self.cname_item_lens = batch_instance[DName.C_LEN]
        self.value_item_lens = batch_instance[DName.V_LEN]
        q_span_tok_lens = batch_instance[DName.Q_SPAN_LEN]
        tname_token_lens = batch_instance[DName.T_TOKS_LEN]
        cname_token_lens = batch_instance[DName.C_TOKS_LEN]
        value_token_lens = batch_instance[DName.V_TOKS_LEN]
        self.all_inputs_name += [input_qtc_src.name, input_qtc_pos.name, input_qtc_sent.name,
                                 input_qtc_mask.name, input_qtc_task.name,
                                 input_qv_src.name, input_qv_pos.name, input_qv_sent.name,
                                 input_qv_mask.name, input_qv_task.name,
                                 input_q_pos.name, input_t_pos.name, input_c_pos.name, input_v_pos.name,
                                 q_span_lens.name, q_span_tok_lens.name,
                                 self.tname_item_lens.name, self.cname_item_lens.name, self.value_item_lens.name,
                                 tname_token_lens.name, cname_token_lens.name, value_token_lens.name]

        config_path = self.encoder_params.get("config_path")
        use_fp16 = self.encoder_params.get("use_fp16", False)
        ernie_config = ErnieConfig(config_path)
        ernie_qtc = ErnieModel(src_ids=input_qtc_src,
                           position_ids=input_qtc_pos,
                           sentence_ids=input_qtc_sent,
                           task_ids=input_qtc_task,
                           input_mask=input_qtc_mask,
                           config=ernie_config,
                           use_fp16=use_fp16)
        qtc_enc_output = ernie_qtc.get_sequence_output()
        qtc_enc_output = layers.fc(qtc_enc_output, size=self.hidden_size, num_flatten_dims=2,
                                   **nn_utils.param_attr('ernie_output', self.init_scale, need_bias=True))
        ernie_qv = ErnieModel(src_ids=input_qv_src,
                           position_ids=input_qv_pos,
                           sentence_ids=input_qv_sent,
                           task_ids=input_qv_task,
                           input_mask=input_qv_mask,
                           config=ernie_config,
                           use_fp16=use_fp16)
        qv_enc_output = ernie_qv.get_sequence_output()
        qv_enc_output = layers.fc(qv_enc_output, size=self.hidden_size, num_flatten_dims=2,
                                  **nn_utils.param_attr('ernie_output', self.init_scale, need_bias=True))

        output_state = layers.dropout(x=ernie_qtc.get_pooled_output() + ernie_qv.get_pooled_output(),
                                     dropout_prob=self.dropout,
                                     dropout_implementation="upscale_in_train")
        output_state = layers.fc(output_state, size=self.hidden_size, num_flatten_dims=1,
                                 **nn_utils.param_attr('all_state', self.init_scale, need_bias=True))
        #question_enc = nn_utils.batch_gather(qtc_enc_output, input_q_pos)
        question_enc, _ = self._table_encoder(
                qtc_enc_output, None, q_span_lens, input_q_pos, q_span_tok_lens,
                self.question_fea_emb, name='question_enc')
        max_q_span_len = input_q_pos.shape[1]
        self.question_mask = layers.sequence_mask(q_span_lens, maxlen=max_q_span_len, dtype='float32')

        q_padding_mask = self.question_mask - 1.0
        table_enc, _ = self._table_encoder(
                qtc_enc_output, None, self.tname_item_lens, input_t_pos, tname_token_lens,
                self.table_fea_emb, name=('table_enc', 'table_enc_attn', 'tab_enc_out'),
                question_encoding=question_enc, q_padding_mask=q_padding_mask)
        column_enc, _ = self._table_encoder(
                qtc_enc_output, None, self.cname_item_lens, input_c_pos, cname_token_lens,
                self.column_fea_emb, name=('table_enc', 'table_enc_attn', 'col_enc_out'),
                question_encoding=question_enc, q_padding_mask=q_padding_mask)
        value_enc, _ = self._table_encoder(
                qv_enc_output, None, self.value_item_lens, input_v_pos, value_token_lens,
                self.value_fea_emb, name=('table_enc', 'table_enc_attn', 'val_enc_out'),
                question_encoding=question_enc, q_padding_mask=q_padding_mask)

        return [[output_state, output_state], question_enc, table_enc, column_enc, value_enc]

    def _read_question(self, question_info):
        """read question info

        Args:
            question_info (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        self.question_ids = question_info[DName.INPUT_IDS]
        self.question_lens = question_info[DName.SEQ_LENS]
        self.q_span_lens = question_info[DName.NAME_LENS]
        self.q_span_pos = question_info[DName.NAME_POS]
        self.q_span_tok_lens = question_info[DName.NAME_TOK_LEN]

        max_question_len = fluid.layers.shape(self.q_span_pos)[1]
        self.question_mask = fluid.layers.sequence_mask(
                self.q_span_lens, maxlen=max_question_len, dtype="float32")

    def _read_table_name(self, tname_info):
        """read table name info

        Args:
            tname_info (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        self.tname_ids = tname_info[DName.INPUT_IDS]
        self.tname_lens = tname_info[DName.SEQ_LENS]
        self.tname_item_lens = tname_info[DName.NAME_LENS]
        self.tname_pos = tname_info[DName.NAME_POS]
        self.tname_token_lens = tname_info[DName.NAME_TOK_LEN]

    def _read_column_name(self, cname_info):
        """read column name info

        Args:
            cname_info (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        self.cname_ids = cname_info[DName.INPUT_IDS]
        self.cname_lens = cname_info[DName.SEQ_LENS]
        self.cname_item_lens = cname_info[DName.NAME_LENS]
        self.cname_pos = cname_info[DName.NAME_POS]
        self.cname_token_lens = cname_info[DName.NAME_TOK_LEN]

    def _read_value(self, value_info):
        """read value info

        Args:
            value_info (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        self.value_ids = value_info[DName.INPUT_IDS]
        self.value_lens = value_info[DName.SEQ_LENS]
        self.value_item_lens = value_info[DName.NAME_LENS]
        self.value_pos = value_info[DName.NAME_POS]
        self.value_token_lens = value_info[DName.NAME_TOK_LEN]

    def _src_embedder(self, tokens):
        """question_embedder.

        Args:
            tokens (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        tokens = layers.squeeze(tokens, axes=[-1])
        param = fluid.ParamAttr(name="src_embedding", initializer=self._emb_weight_init)
        output = fluid.embedding(
                    input=tokens,
                    size=[self.src_vocab_size, self.embedding_size],
                    dtype='float32',
                    param_attr=param,
                    is_sparse=False)
        if not self.finetune_emb:
            output.stop_gradient = True
        return output

    def _lf_embedder(self, tokens, token_lens=None):
        """lf embedder.

        Args:
            tokens (Variable): [batch_size, seq_len]
            token_lens (Variable): Default is None.

        Returns: TODO

        Raises: NULL

        """
        self._batch_size = layers.shape(self.question_encoding)[0]

        ##  Grammar Rule Embedding
        self._grammar_vocab = tensor.cast(tensor.assign(self.grammar.gmr_vocab.astype(np.int32)), dtype='int64')
        self._grammar_emb = fluid.embedding(
                    input=self._grammar_vocab,
                    size=[self.grammar.grammar_size, self.lf_emb_size],
                    dtype='float32',
                    is_sparse=False,
                    param_attr=fluid.ParamAttr(name="lf_embedding",
                                               initializer=nn_utils.uniform(self.init_scale)))

        batch_emb_lookup_grammar = layers.expand(layers.unsqueeze(self._grammar_emb, [0]),
                                                 [self._batch_size, 1, 1])
        def _table_to_lf_input(ori_encoding):
            """trans ori_encoding to size of lf_embedding
            """
            output = layers.fc(input=ori_encoding, size=self.lf_emb_size, num_flatten_dims=2,
                               **nn_utils.param_attr('fc_table2lf_input', self.init_scale, need_bias=False))
            return output
        batch_emb_lookup_all = tensor.concat([batch_emb_lookup_grammar,
                                              _table_to_lf_input(self.tname_encoding),
                                              _table_to_lf_input(self.cname_encoding),
                                              _table_to_lf_input(self.value_encoding)], axis=1)
        lf_embedding = nn_utils.batch_gather_2d(batch_emb_lookup_all, tokens)

        ## Grammar Rule 类型 Embedding
        self._grammar2name = layers.cast(layers.assign(self.grammar.gmr2name_arr.astype(np.int32)), dtype='int64')
        lf_name = layers.reshape(layers.gather(self._grammar2name, layers.reshape(tokens, shape=[-1])),
                                 shape=tokens.shape)
        lf_name.stop_gradient = True
        lf_name_emb = fluid.embedding(
                    input=lf_name,
                    size=[self.grammar.name_size, self.lf_name_emb_size],
                    dtype='float32',
                    is_sparse=False,
                    param_attr=fluid.ParamAttr(name="lf_name_embedding",
                                               initializer=nn_utils.uniform(self.init_scale)))

        output = layers.concat([lf_embedding, lf_name_emb], axis=-1)
        if token_lens is not None:
            mask = layers.sequence_mask(token_lens, maxlen=layers.shape(tokens)[1], dtype='float32')
            output = layers.elementwise_mul(output, mask, axis=0)
        return output

    def _feature_embedder(self, one_hot_fea, name):
        """feature embedder

        Args:
            one_hot_fea (Variable): shape=[batch_size, feature_dim], dtype=float32
            name (str): layers name

        Returns: TODO

        Raises: NULL
        """
        output = layers.fc(input=one_hot_fea, size=self.hidden_size, num_flatten_dims=2,
                           **nn_utils.param_attr(name, self.init_scale, need_bias=True))
        return output

    def _question_encoder(self):
        """question_encoder.

        Returns: TODO

        Raises: NULL

        """
        encoder_2d = models.Sequence2DEncoder(
                'simple_sum', dropout=self.dropout, init_scale=self.init_scale, name='question_span')
        q_enc_tmp, _ = encoder_2d.forward(self.question_emb, self.question_lens,
                self.q_span_lens, self.q_span_pos, self.q_span_tok_lens)

        if self.question_fea_emb is not None:
            q_enc_tmp_with_fea = layers.elementwise_add(q_enc_tmp, self.question_fea_emb)
        else:
            q_enc_tmp_with_fea = q_enc_tmp
        question_encoder = models.RNNEncoder(self.encoder_layers, self.hidden_size // 2, bidirectional=True,
                                  dropout=self.dropout, init_scale=self.init_scale, name='question_enc_rnn')
        enc_output, enc_final_state = question_encoder.forward(q_enc_tmp_with_fea, self.q_span_lens)
        return enc_output, enc_final_state

    def _table_encoder(self, inputs, input_lens, name_lens, name_pos, name_tok_len, inputs_fea,
                             name, question_encoding=None, q_padding_mask=None):
        """table encoder.

        Args:
            inputs (TYPE): NULL
            input_lens (TYPE): NULL
            name_lens (TYPE): NULL
            name_pos (TYPE): NULL
            name_tok_len (TYPE): NULL
            inputs_fea (TYPE): NULL
            name (str/list): NULL
            question_encoding(Variable): NULL
            q_padding_mask(Variable): NULL

        Returns: TODO

        Raises: NULL

        """
        if type(name) is tuple or type(name) is list:
            assert len(name) == 3, "name tuple's len must equal to 3"
            enc_name, attn_name, fc_name = name
        else:       # type(name) is str
            enc_name = name + '_rnn'
            attn_name = name + '_attn'
            fc_name = name + '_out_fc'

        if self.table_enc_type == 'birnn':
            encoder = models.Sequence2DEncoder(
                    self.table_enc_type, dropout=self.dropout, init_scale=self.init_scale, name=enc_name,
                    num_layers=self.encoder_layers, hidden_size=self.hidden_size // 2, bidirectional=True)
        elif self.table_enc_type == 'simple_sum':
            encoder = models.Sequence2DEncoder(
                    self.table_enc_type, dropout=self.dropout, init_scale=self.init_scale, name=name)
        else:
            raise ValueError("unsupported table encoder type: %s" % (self.table_enc_type))

        enc_output, _ = encoder.forward(inputs, input_lens, name_lens, name_pos, name_tok_len)
        if self.table_attention is not None and question_encoding is not None:
            attn = models.Attention(score_type=self.table_attention, name=attn_name)
            ctx = attn.forward(enc_output, question_encoding, padding_mask=q_padding_mask)
            #enc_output_attn = layers.elementwise_add(enc_output, ctx)
            enc_output = layers.concat([enc_output, ctx], axis=-1)
        if inputs_fea is not None:
            #enc_output = layers.elementwise_add(enc_output, inputs_fea)
            enc_output = layers.concat([enc_output, inputs_fea], axis=-1)

        final_output = layers.fc(enc_output, size=self.hidden_size, num_flatten_dims=2,
                                 **nn_utils.param_attr(fc_name, self.init_scale, need_bias=True))

        return final_output, None

    def _decoder(self, enc_output, enc_state, mode="train", beam_size=1):
        """decoder

        Args:
            enc_output (TYPE): NULL
            enc_state (TYPE): NULL
            mode (string): running mode: train|infer. default is "train"
            beam_size (int): default is 1

        Returns: TODO

        Raises: NULL

        """
        output_layer = functools.partial(gmr_models.grammar_output, name='decoder_output')
        decode_cell = models.RNNDecodeCell(self.hidden_size, dropout=self.dropout, init_scale=self.init_scale)
        dec_vocab = gmr_models.DecoderDynamicVocab(self.tname_encoding, self.tname_item_lens,
                                                   self.cname_encoding, self.cname_item_lens,
                                                   self.value_encoding, self.value_item_lens,
                                                   self.column2table_mask)
        dec_attn_key = layers.fc(self.question_encoding, size=self.hidden_size, num_flatten_dims=2,
                                 **nn_utils.param_attr('dec_attn_key', self.init_scale, need_bias=True))

        init_state0 = layers.fc(enc_state[0], size=self.hidden_size, num_flatten_dims=1, act='tanh',
                                **nn_utils.param_attr('dec_init_state0_fc', self.init_scale, need_bias=True))
        init_state1 = layers.fc(enc_state[1], size=self.hidden_size, num_flatten_dims=1, act='tanh',
                                **nn_utils.param_attr('dec_init_state1_fc', self.init_scale, need_bias=True))

        #dec_init_zero = layers.zeros_like(init_state1)
        init_state = [
            decode_cell.get_initial_states(batch_ref=self.question_encoding, shape=[self.hidden_size]),
            [init_state0, init_state1],
        ]
        dec_cell_params = {
                    "attn_k": dec_attn_key,
                    "attn_v": self.question_encoding,
                    "padding_mask": self.question_mask - 1.0
                }
        if mode == "train":
            ## 解码端词表 emb ##
            self.train_label_emb = self._lf_embedder(self.train_label, self.label_lens)
            dec_output, dec_state = fluid.layers.rnn(cell=decode_cell,
                                        inputs=self.train_label_emb,
                                        initial_states=init_state,
                                        sequence_length=None,
                                        **dec_cell_params)

            outputs, _ = output_layer(dec_output,
                                      self.infer_actions,
                                      self.infer_gmr_mask,
                                      self.valid_table_mask,
                                      dec_vocab,
                                      self.grammar)
            return layers.elementwise_mul(outputs, self.label_mask, axis=0)
        elif mode == "infer":
            gmr_infer_decoder = gmr_models.GrammarInferDecoder(decode_cell,
                                        beam_size=self.beam_size,
                                        grammar=self.grammar,
                                        fn_embedding=self._lf_embedder,
                                        fn_output=output_layer)

            outputs, _ = gmr_models.decode_with_grammar(
                                gmr_infer_decoder,
                                inits=init_state,
                                decode_vocab=dec_vocab,
                                max_step_num=self.max_infer_step,
                                **dec_cell_params)
            return outputs
        else:
            raise ValueError("unsupported running mode: %s" % (mode))

    def _calc_loss(self, predictions, label, loss_type='cross_entropy'):
        """calc loss

        Args:
            predictions (TYPE): NULL
            label (TYPE): NULL
            loss_type (str): cross_entropy|softmax_ce|predict_prob

        Returns: TODO

        Raises: NULL
        """
        if loss_type == 'softmax_ce':
            loss_tmp = self._calc_softmax_cross_entropy(predictions, label)
        elif loss_type == 'cross_entropy':
            loss_tmp = self._calc_cross_entropy(predictions, label)
        else:
            raise ValueError('unsupported loss type: %s' % (loss_type))

        max_len = fluid.layers.shape(label)[1]
        target_mask = fluid.layers.sequence_mask(self.label_lens, maxlen=max_len, dtype='float32')
        loss = loss_tmp * target_mask
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        return loss

    def _calc_softmax_cross_entropy(self, predictions, label):
        """calc cross entropy loss

        Args:
            predictions (Tensor): NULL
            label (Tensor): NULL

        Returns: Tensor

        Raises: NULL

        """
        label_for_loss = layers.unsqueeze(label, [2])
        ce_loss = fluid.layers.softmax_with_cross_entropy(logits=predictions, label=label_for_loss, soft_label=False)
        ce_loss = fluid.layers.unsqueeze(ce_loss, axes=[2])

        return ce_loss

    def _calc_cross_entropy(self, predictions, label):
        """calc cross entropy loss

        Args:
            predictions (Tensor): NULL
            label (Tensor): NULL

        Returns: Tensor

        Raises: NULL

        """
        label_for_loss = layers.unsqueeze(label, [2])
        ce_loss = fluid.layers.cross_entropy(input=predictions, label=label_for_loss, soft_label=False)
        ce_loss = fluid.layers.unsqueeze(ce_loss, axes=[2])

        return ce_loss


if __name__ == "__main__":
    """run some simple test cases"""
    pass

