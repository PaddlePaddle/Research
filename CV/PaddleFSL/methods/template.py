import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

class Template(fluid.dygraph.Layer):
    def __init__(self, args):

        super(Template, self).__init__()

        self.args = args
        if args.backbone == 'Conv4':
            from models.conv4 import Conv4
            self.encoder = Conv4(self.args)
        elif args.backbone == 'Resnet12':
            from models.resnet12 import ResNet12
            self.encoder = ResNet12(self.args)

    def split_train_query(self, x, y):
        new_shape = [self.args.n_way, self.args.k_shot+self.args.n_query] + x.shape[1:]
        x_reshape = fluid.layers.reshape(x, new_shape)  # [n_way,(k_shot+n_query),32,32,3]
        x_train = x_reshape[:,:self.args.k_shot]  # [n_way,k_shot,32,32,3]
        x_query = x_reshape[:,self.args.k_shot:]  # [n_way,n_query,32,32,3]

        y_reshape = fluid.layers.reshape(y, [self.args.n_way, self.args.k_shot+self.args.n_query])
        y_train = fluid.layers.reshape(y_reshape[:,:self.args.k_shot], [-1,1])
        y_query = fluid.layers.reshape(y_reshape[:,self.args.k_shot:], [-1,1])  # [n_way*n_query,1]

        return x_train, x_query, y_train, y_query
        
    def forward(self, samples, labels):
        embeddings = self.encoder(samples)
        predicted_query_logits, query_label = self.specific_method(embeddings, labels)

        return predicted_query_logits, query_label

    def loss(self, x, y):
        query_logits, query_labels = self.forward(x, y)
        loss = fluid.layers.softmax_with_cross_entropy(query_logits, query_labels)
        pred = fluid.layers.softmax(query_logits) # [75,5]
        acc = fluid.layers.accuracy(pred, query_labels)
        return loss, acc

    def specific_method(self, x, y):
        raise NotImplementedError('To be implemented by subclass')
        


    
