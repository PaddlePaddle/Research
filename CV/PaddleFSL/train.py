import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from visualdl import LogWriter
from utils import parse_args, prepare_model, prepare_optimizer, prepare_dataloader

import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
code_flag = os.system('nvidia-smi')
cuda_exist = code_flag == 0

def train():
    args = parse_args()
    print(args)

    if args.use_gpu and cuda_exist:
        place = fluid.CUDAPlace(0)
        print('GPU is used...')
    else:
        place = fluid.CPUPlace()
        print('CPU is used...')


    with fluid.dygraph.guard(place):
        print('start training ... ')

        # prepare method
        model = prepare_model(args)
        model.train()

        # prepare optimizer
        opt = prepare_optimizer(args, model)
        
        # prepare dataloader
        train_data_batches, val_data_batches = prepare_dataloader(args)

        best_val_acc = 0
        with LogWriter(logdir=args.log_dir+'train/') as writer:
            for epoch in range(args.epochs):
                train_loss, train_acc = [], []
                for batch_id, batch in enumerate(train_data_batches):
                    samples, label = batch
                    samples = fluid.dygraph.to_variable(samples)
                    labels = fluid.dygraph.to_variable(label)
                    loss, acc = model.loss(samples, labels)
                    avg_loss = fluid.layers.mean(loss)
                    train_loss.append(avg_loss.numpy())
                    train_acc.append(acc.numpy())

                    if batch_id % 100 == 0:
                        print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, avg_loss.numpy(), acc.numpy()))
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()
                
                writer.add_scalar(tag="train_loss", step=epoch, value=np.mean(train_loss))
                writer.add_scalar(tag="train_acc", step=epoch, value=np.mean(train_acc))

                model.eval()
                accuracies = []
                losses = []
                for batch_id, batch in enumerate(val_data_batches):
                    samples, label = batch
                    samples = fluid.dygraph.to_variable(samples)
                    labels = fluid.dygraph.to_variable(label)
                    loss, acc = model.loss(samples, labels)
                    avg_loss = fluid.layers.mean(loss)
                    accuracies.append(acc.numpy())
                    losses.append(avg_loss.numpy())
                acc_avg_val = np.mean(accuracies)
                print("[validation] accuracy/loss: {}/{}".format(acc_avg_val, np.mean(losses)))
                model.train()

                writer.add_scalar(tag="val_loss", step=epoch, value=np.mean(losses))
                writer.add_scalar(tag="val_acc", step=epoch, value=acc_avg_val)

                if acc_avg_val > best_val_acc:
                    # save params of model
                    fluid.save_dygraph(model.state_dict(), args.log_dir+args.method+'_'+args.backbone+'_'+str(args.k_shot)+'shot_'+str(args.n_way)+'way'+str(epoch))
                    # save optimizer state
                    # fluid.save_dygraph(opt.state_dict(), args.log_dir+args.method+'_'+args.backbone+'_'+str(args.k_shot)+'shot_'+str(args.n_way)+'way'+str(epoch))
                    best_val_acc = acc_avg_val

if __name__ == "__main__":
    train()


