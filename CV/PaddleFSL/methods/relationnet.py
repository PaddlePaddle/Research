import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from methods.template import Template
from utils import euclidean_dis, cosine_dis

class RelationNet(Template):
    def __init__(self, args):
        super(RelationNet, self).__init__(args)
        self.reslation_module = Relation_module(self.args)
        self.mse_loss = fluid.dygraph.MSELoss(reduction='sum')

    def specific_method(self, embeddings, labels):
        x_train, x_query, y_train, y_query = self.split_train_query(embeddings, labels)
        protos = fluid.layers.reduce_mean(x_train, dim=1) # [5,64,21,21]
        query_embs = fluid.layers.reshape(x_query, [self.args.n_way*self.args.n_query, x_query.shape[-3],x_query.shape[-2], x_query.shape[-1]]) #[75,64,21,21]
        # concatenate
        protos_exp = fluid.layers.expand(fluid.layers.unsqueeze(protos,[0]), [query_embs.shape[0],1,1,1,1])
        query_embs_exp = fluid.layers.expand(fluid.layers.unsqueeze(query_embs,[1]), [1,protos.shape[0],1,1,1])
        concat_pairs = fluid.layers.concat(input=[protos_exp, query_embs_exp], axis=2)
        concat_paris_all = fluid.layers.reshape(concat_pairs, [concat_pairs.shape[0]*concat_pairs.shape[1],concat_pairs.shape[2],concat_pairs.shape[3],concat_pairs.shape[4]])  # [75*5, 128,21,21]

        relation_scores = fluid.layers.reshape(self.reslation_module(concat_paris_all), [concat_pairs.shape[0], concat_pairs.shape[1]])  # [75,5]
        return relation_scores, y_query
    
    def loss(self, x, y):
        relation_scores, query_labels = self.forward(x, y)
        query_labels_onehot = fluid.layers.one_hot(query_labels, depth=self.args.n_way)
        if self.args.backbone == 'Conv4':
            loss = self.mse_loss(relation_scores, query_labels_onehot)
        elif self.args.backbone == 'Resnet12':
            relation_scores = fluid.layers.reshape(relation_scores, [self.args.n_way*self.args.n_query, self.args.n_way])
            loss = fluid.layers.softmax_with_cross_entropy(relation_scores, query_labels)
        pred = fluid.layers.softmax(relation_scores) # [75,5]
        acc = fluid.layers.accuracy(pred, query_labels)
        return loss, acc

class Conv_block(fluid.dygraph.Layer):
    def __init__(self, num_channels=64, num_filters=64, padding=0, pooltype=None):
        
        super(Conv_block, self).__init__()
        self.conv = Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=1, padding=padding)
        self.batch_norm = BatchNorm(num_filters, act='relu')
        self.pooling = Pool2D(pool_size=2, pool_stride=2, pool_type=pooltype)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.pooling(x)
        return x

class Relation_module(fluid.dygraph.Layer):
    def __init__(self, args):
        super(Relation_module, self).__init__()
        self.args = args
        if 'imagenet' in self.args.dataset or 'cub' in self.args.dataset:
            linear_dim = 3*3*self.args.num_filters
        else:
            linear_dim = self.args.num_filters
        inp_channels = self.args.num_filters*2 if self.args.backbone=='Conv4' else self.args.resnet12_num_filters[-1]*2
        padding = 1 if self.args.dataset=='omniglot' else 0
        self.conv0 = Conv_block(num_channels=inp_channels, num_filters=64, padding=padding, pooltype=self.args.pooling_type)
        self.conv1 = Conv_block(num_channels=64, num_filters=64, padding=padding, pooltype=self.args.pooling_type)
        self.fc0 = Linear(linear_dim, 8, act='relu')
        self.fc1 = Linear(8, 1)
    
    def forward(self, inputs):
        # print(inputs.shape)
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = fluid.layers.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        if self.args.backbone == 'Conv4':
            x = fluid.layers.sigmoid(x)
        return x