# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle as P
from paddle import nn
from paddle.nn import functional as F

from paddlenlp.transformers import ErniePretrainedModel

class ErnieModelForDuIE(ErniePretrainedModel):
    def __init__(self, ernie, num_classes=112):
        super(ErnieModelForDuIE, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow bert to be config
        self.dropout = nn.Dropout(self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                labels,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')
        labels = P.cast(labels, dtype='float32')
        loss = bce_logit_loss(logits, labels)
        mask = P.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = P.sum(loss.mean(axis=2), axis=1) / P.sum(mask, axis=1)
        loss = loss.mean()

        seq_len = P.cast(P.sum(mask, axis=1), dtype='int16')
        logits = F.sigmoid(logits)
        return loss, logits, seq_len