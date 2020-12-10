import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from methods.template import Template
from utils import euclidean_dis, cosine_dis

class ProtoNet(Template):
    def __init__(self, args):
        super(ProtoNet, self).__init__(args)

    def specific_method(self, embeddings, labels):
        embeddings_flat = fluid.layers.flatten(embeddings)
        x_train, x_query, y_train, y_query = self.split_train_query(embeddings_flat, labels)
        protos = fluid.layers.reduce_mean(x_train, dim=1)
        query_embs = fluid.layers.reshape(x_query, [self.args.n_way*self.args.n_query, -1])
        predicted_query_logits = euclidean_dis(protos, query_embs) / self.args.temperature
        return predicted_query_logits, y_query