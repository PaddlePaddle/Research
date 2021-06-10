#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import paddle


class LSTMAttentionModel(object):
    """LSTM Attention Model"""

    def __init__(self,
                 bias_attr,
                 embedding_size=512,
                 lstm_size=1024,
                 drop_rate=0.5):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

    def forward(self, input, is_training):
        input_fc = paddle.static.nn.fc(
            x=input,
            size=self.embedding_size,
            activation='tanh',
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(coeff=0.0),
                initializer=paddle.nn.initializer.Normal(std=0.0)),
            name='rgb_fc')

        lstm_forward_fc = paddle.static.nn.fc(
            x=input_fc,
            size=self.lstm_size * 4,
            activation=None,
            bias_attr=False,  # video_tag
            name='rgb_fc_forward')

        lstm_forward, _ = paddle.fluid.layers.dynamic_lstm(
            input=lstm_forward_fc,
            size=self.lstm_size * 4,
            is_reverse=False,
            name='rgb_lstm_forward')

        lsmt_backward_fc = paddle.static.nn.fc(
            x=input_fc,
            size=self.lstm_size * 4,
            activation=None,
            bias_attr=False,  #video_tag
            name='rgb_fc_backward')

        lstm_backward, _ = paddle.fluid.layers.dynamic_lstm(
            input=lsmt_backward_fc,
            size=self.lstm_size * 4,
            is_reverse=True,
            name='rgb_lstm_backward')

        lstm_concat = paddle.concat(x=[lstm_forward, lstm_backward], axis=1)

        lstm_dropout = paddle.fluid.layers.nn.dropout(
            x=lstm_concat,
            dropout_prob=self.drop_rate,
            is_test=(not is_training))

        lstm_weight = paddle.static.nn.fc(
            x=lstm_dropout,
            size=1,
            activation='sequence_softmax',
            bias_attr=False,  #video_tag
            name='rgb_weight')

        scaled = paddle.multiply(x=lstm_dropout, y=lstm_weight)
        lstm_pool = paddle.fluid.layers.sequence_pool(
            input=scaled, pool_type='sum')

        return lstm_pool
