import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from dataloader.cifarfs import LoadData
from dataloader.data_sampler import BatchSampler
from utils import parse_args, prepare_model, prepare_optimizer, prepare_dataloader

import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
code_flag = os.system('nvidia-smi')
cuda_exist = code_flag == 0

def test():
    args = parse_args()
    print(args)

    if args.use_gpu and cuda_exist:
        place = fluid.CUDAPlace(0)
        print('GPU is used...')
    else:
        place = fluid.CPUPlace()
        print('CPU is used...')

    with fluid.dygraph.guard(place):
        print('start testing ... ')

        # prepare method
        model = prepare_model(args)

        # load checkpoint
        # model_dict, _ = fluid.dygraph.load_persistables("log/")
        params_dict, opt_dict = fluid.load_dygraph(args.log_dir+'checkpoint/'+args.dataset+'/'+args.method+'_'+args.backbone+'_'+str(args.k_shot)+'shot_'+str(args.n_way)+'way')
        model.load_dict(params_dict)
        print("checkpoint loaded")

        # prepare optimizer
        opt = prepare_optimizer(args, model)
        
        # prepare dataloader
        test_data_batches = prepare_dataloader(args)
        
        model.eval()
        accuracies = []
        losses = []
        for batch_id, batch in enumerate(test_data_batches):
            samples, label = batch
            samples = fluid.dygraph.to_variable(samples)
            labels = fluid.dygraph.to_variable(label)
            loss, acc = model.loss(samples, labels)
            avg_loss = fluid.layers.mean(loss)
            accuracies.append(acc.numpy())

        mean = np.mean(accuracies)
        stds = np.std(accuracies)
        ci95 = 1.96*stds/np.sqrt(args.test_episodes)
        print("meta-testing accuracy: {}, 95_confidence_interval: {}".format(mean, ci95))

if __name__ == "__main__":
    test()


