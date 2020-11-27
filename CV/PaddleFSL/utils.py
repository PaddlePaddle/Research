import numpy as np
import paddle
import paddle.fluid as fluid
import argparse
from dataloader.data_sampler import BatchSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='miniimagenet', type=str, help='miniimagenet/omniglot/cifarfs/fc100/cub/tieredimagenet')
    parser.add_argument('--backbone', default='Conv4', type=str, help='model: Conv4/Resnet12')
    parser.add_argument('--method', default='protonet', type=str, help='protonet/relationnet')
    parser.add_argument('--log_dir', default='./logs/', type=str, help='Directory where to write event logs and checkpoints')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--episodes', default=1000, type=int, help='number of episodes per epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default=False, type=bool, help='if using learning rate scheduler')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--if_dropout', default=False, type=bool, help='if using dropout in backbone')
    parser.add_argument('--n_way', default=5, type=int, help='number of classes in a task')
    parser.add_argument('--k_shot', default=1, type=int, help='number of training sample per class')
    parser.add_argument('--n_query', default=15, type=int, help='number of queries per class')
    parser.add_argument('--train_aug', default=False, type=bool, help='perform data augmentation or not during training')
    parser.add_argument('--meta_batch', default=1, type=int, help='number of meta batch')

    parser.add_argument('--use_gpu', default=1, type=int, help='whether gpu is used')
    # for backbones
    parser.add_argument('--num_filters', default=64, type=int, help='number of queries per class')
    parser.add_argument('--pooling_type', default='max', type=str, help='max/avg')
    parser.add_argument('--resnet12_num_filters', nargs='+', default=[64,128,256,512], type=int, help='number of conv channels in ResNet12 backbone, eg. --resnet12_num_filters 64 128 256 512') 

    # testing
    parser.add_argument('--test_mode', default=False, type=bool, help='if in test mode')
    parser.add_argument('--test_model_epoch', default=1, type=int, help='test model epoch')
    parser.add_argument('--test_episodes', default=600, type=int, help='number of testing episodes')

    # for Protonet
    parser.add_argument('--distance_metric', default='euclidean', type=str, help='euclidean/cosine distance for protonet')
    parser.add_argument('--temperature', default=1.0, type=float, help='distance temperature for protonet')

    return parser.parse_args()

def prepare_model(args):
    if args.method == 'protonet':
        from methods.protonet import ProtoNet
        model = ProtoNet(args)
    elif args.method == 'relationnet':
        from methods.relationnet import RelationNet
        model = RelationNet(args)
    return model

def prepare_optimizer(args, model):
    if args.backbone == 'Conv4':
        if not args.lr_scheduler:
            opt = fluid.optimizer.Adam(learning_rate=args.lr, parameter_list=model.parameters())
        else:
            opt = fluid.optimizer.Adam(learning_rate=fluid.layers.exponential_decay(learning_rate=args.lr, decay_steps=(args.epochs*args.episodes)//5, decay_rate=0.5, staircase=True), parameter_list=model.parameters()) 
    elif args.backbone == 'Resnet12':
        if not args.lr_scheduler:
            opt = fluid.optimizer.MomentumOptimizer(learning_rate=args.lr, momentum=0.9, use_nesterov=True, parameter_list=model.parameters(), regularization=fluid.regularizer.L2Decay(regularization_coeff=args.weight_decay))
        else:
            opt = fluid.optimizer.MomentumOptimizer(learning_rate=fluid.layers.exponential_decay(learning_rate=args.lr, decay_steps=(args.epochs*args.episodes)//4, decay_rate=0.1, staircase=True), momentum=0.9, use_nesterov=True, parameter_list=model.parameters(), regularization=fluid.regularizer.L2Decay(regularization_coeff=args.weight_decay))
    return opt

def prepare_dataloader(args):
    if args.dataset == 'cifarfs':
        from dataloader.cifarfs import LoadData
    elif args.dataset == 'miniimagenet':
        from dataloader.miniimagenet import LoadData
    elif args.dataset == 'tieredimagenet':
        from dataloader.tieredimagenet import LoadData
    elif args.dataset == 'cub':
        from dataloader.cub import LoadData
    elif args.dataset == 'omniglot':
        from dataloader.omniglot import LoadData
    elif args.dataset == 'fc100':
        from dataloader.fc100 import LoadData
    # create dataloader
    if not args.test_mode:
        train_data = LoadData('train')
        train_data_batches = BatchSampler(data=train_data, args=args)

        val_data = LoadData('val')
        val_data_batches = BatchSampler(data=val_data, args=args)
        return train_data_batches, val_data_batches
    else:
        test_data = LoadData('test')
        test_data_batches = BatchSampler(data=test_data, args=args)
        return test_data_batches

def split_batch(batch, args):
    batch_data, batch_label = batch
    data_reshape = fluid.layers.reshape(batch_data, [args.n_way,(args.k_shot+args.n_query),batch_data.shape[-3],batch_data.shape[-2],batch_data.shape[-1]]) # [5,7,32,32,3]
    label_reshape = fluid.layers.reshape(batch_data, [args.n_way,(args.k_shot+args.n_query)]) # [5,7]
    train_data = data_reshape[:,:args.k_shot,:,:,:]
    query_data = data_reshape[:,args.k_shot:,:,:,:]
    train_label = data_reshape[:,:args.k_shot]
    query_label = data_reshape[:,args.k_shot:]
    return train_data, train_label, query_data, query_label

def euclidean_dis(x, y):
    x_exp = fluid.layers.expand(fluid.layers.unsqueeze(x,[0]), [y.shape[0],1,1])
    y_exp = fluid.layers.expand(fluid.layers.unsqueeze(y,[1]), [1,x.shape[0],1])
    output = - fluid.layers.reduce_mean((x_exp-y_exp)**2, dim=-1)
    return output

def cosine_dis(x, y):
    x_exp = fluid.layers.expand(fluid.layers.unsqueeze(x,[0]), [y.shape[0],1,1])
    y_exp = fluid.layers.expand(fluid.layers.unsqueeze(y,[1]), [1,x.shape[0],1])
    x_norm = fluid.layers.l2_normalize(x, axis=-1)
    y_norm = fluid.layers.l2_normalize(y, axis=-1)
    output = fluid.layers.reduce_sum(x_norm*y_norm, dim=-1)
    return output
