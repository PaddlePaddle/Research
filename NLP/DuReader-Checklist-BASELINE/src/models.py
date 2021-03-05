from paddlenlp.transformers import ErniePretrainedModel, BertPretrainedModel, RobertaPretrainedModel
from paddle import nn
import paddle

class ErnieForQuestionAnswering(ErniePretrainedModel):
    def __init__(self, ernie):
        super(ErnieForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits

class BertForQuestionAnswering(BertPretrainedModel):
    def __init__(self, bert):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = bert  # allow bert to be config
        self.classifier = nn.Linear(self.bert.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.bert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits

class RobertaForQuestionAnswering(RobertaPretrainedModel):
    def __init__(self, roberta):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits