import paddle
from paddle import nn
from paddle import optimizer
from paddle.nn import functional as F
from paddle import optimizer
import numpy as np
import paddle
try:
    from model.learner import Learner
except:
    from model.learner import Learner
from copy import deepcopy


class Meta(nn.Layer):
    """
    CHAML Meta Learner.
    """

    def __init__(self, config):
        super(Meta, self).__init__()
        self.update_lr = config['update_lr']  # task-level inner update learning rate
        self.meta_lr = config['meta_lr']  # meta-level outer learning rate
        self.update_step = config['update_step']  # task-level inner update steps
        self.update_step_test = config['update_step_test']  # update steps for finetunning
        self.LOCAL_FIX_VAR = config['local_fix_var']  # vars[0:LOCAL_FIX_VAR] should be fixed in the local update (fast_weights only update [LOCAL_FIX_VAR:])
        self.sample_batch_size = config['sample_batch_size']  # batch size of samples to feed into Learner
        self.net = Learner(config)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1.0 / 2)
        clip_coef = max_norm / (total_norm + 1e-06)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, poiid_embs=None,
        cont_feat_scalers=None):
        x_uid_spt, x_hist_spt, x_candi_spt = x_spt
        x_uid_qry, x_hist_qry, x_candi_qry = x_qry
        task_num = len(x_uid_spt)
        querysz = len(x_uid_qry[0])  # number of qry set samples in one task
        losses_q = [(0) for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [(0) for _ in range(self.update_step + 1)]  # corrects[i] is the number of correctly-predicted qry samples on step i (for calculating acc)
        task_level_acc = [(0) for _ in range(task_num)]
        task_sample_level_corrects = [[] for _ in range(task_num)]
        for i in range(task_num):
            if cont_feat_scalers is not None:
                scaler = cont_feat_scalers[i]
            else:
                scaler = None

            # find the poi_emb_w of each task
            if poiid_embs is not None:
                self.net.parameters()[0] = paddle.create_parameter(shape=\
                    poiid_embs[i].shape, dtype=str(poiid_embs[i].numpy().
                    dtype), default_initializer=paddle.nn.initializer.
                    Assign(poiid_embs[i]))
                self.net.parameters()[0].stop_gradient = False
            
            # run the i-th task and compute training loss (on support) for k=0
            logits = self.net(x_uid_spt[i], x_hist_spt[i], x_candi_spt[i],
                vars=None, scaler=scaler)
            loss = F.cross_entropy(logits, y_spt[i])
            
            grad = paddle.grad(loss, list(self.net.parameters()))
            fast_weights = list(self.net.parameters())[:self.LOCAL_FIX_VAR] + \
                list(map(lambda p: p[1] - self.update_lr * p[0], zip(
                grad[self.LOCAL_FIX_VAR:], list(self.net.parameters())[self.
                LOCAL_FIX_VAR:])))

            # this is the loss and accuracy before first update
            with paddle.no_grad():
                logits_q = self.net(x_uid_qry[i], x_hist_qry[i],
                    x_candi_qry[i], self.net.parameters(), scaler=scaler)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, axis=-1).argmax(axis=-1)
                correct = paddle.equal(pred_q, y_qry[i]).numpy().sum()
                corrects[0] = corrects[0] + correct
            
            # this is the loss and accuracy after the first update
            logits_q = self.net(x_uid_qry[i], x_hist_qry[i], x_candi_qry[i],
                fast_weights, scaler=scaler)
            loss_q = F.cross_entropy(logits_q, y_qry[i])
            losses_q[1] += loss_q
            with paddle.no_grad():
                pred_q = F.softmax(logits_q, axis=-1).argmax(axis=-1)
                correct = paddle.equal(pred_q, y_qry[i]).numpy().sum()
                corrects[1] = corrects[1] + correct
                if self.update_step == 1:
                    task_sample_level_corrects[i] = paddle.equal(pred_q,
                        y_qry[i]).numpy().tolist()
                    task_level_acc[i] = correct
            
            # run the i-th task and compute loss for k=1~K-1
            for k in range(1, self.update_step):
                logits = self.net(x_uid_spt[i], x_hist_spt[i], x_candi_spt[
                    i], fast_weights, scaler=scaler)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = paddle.grad(loss, fast_weights)
                fast_weights = list(fast_weights)[:self.LOCAL_FIX_VAR] + list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad[
                    self.LOCAL_FIX_VAR:], fast_weights[self.LOCAL_FIX_VAR:])))
                logits_q = self.net(x_uid_qry[i], x_hist_qry[i],
                    x_candi_qry[i], fast_weights, scaler=scaler)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                with paddle.no_grad():
                    pred_q = F.softmax(logits_q, axis=-1).argmax(axis=-1)
                    correct = paddle.equal(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
                    if k == self.update_step - 1:
                        task_sample_level_corrects[i] = paddle.equal(pred_q,
                            y_qry[i]).numpy().tolist()
                        task_level_acc[i] = correct
        
        # end of all tasks, sum over all losses on query set across all tasks
        loss_q_final = losses_q[-1] / task_num
        accs = np.array(corrects) / (querysz * task_num)
        task_level_acc = np.array(task_level_acc) / querysz
        results = {'task_level_acc': task_level_acc,
            'task_sample_level_corrects': task_sample_level_corrects}
        return accs, loss_q_final, results

    def finetuning_adapt(self, x_spt, y_spt, poiid_emb, scaler=None,
        meta_feature=None):
        x_uid_spt, x_hist_spt, x_candi_spt = x_spt
        net = deepcopy(self.net)
        net.parameters()[0] = paddle.create_parameter(shape=poiid_emb.shape,
            dtype=str(poiid_emb.numpy().dtype), default_initializer=paddle.
            nn.initializer.Assign(poiid_emb))
        net.parameters()[0].stop_gradient = False
        if self.update_step_test == 0:
            fast_weights = list(net.parameters())
            del net
            print('>>>>>>>>> no inner loop in meta testing.')
            return fast_weights
        logits = net(x_uid_spt, x_hist_spt, x_candi_spt, scaler=scaler)
        loss = F.cross_entropy(logits, y_spt)
        grad = paddle.grad(loss, list(net.parameters()))
        fast_weights = list(net.parameters())[:self.LOCAL_FIX_VAR] + list(map
            (lambda p: p[1] - self.update_lr * p[0], zip(grad[self.
            LOCAL_FIX_VAR:], list(net.parameters())[self.LOCAL_FIX_VAR:])))
        for k in range(1, self.update_step_test):
            logits = net(x_uid_spt, x_hist_spt, x_candi_spt, fast_weights,
                scaler=scaler)
            loss = F.cross_entropy(logits, y_spt)
            grad = paddle.grad(loss, fast_weights)
            fast_weights = list(fast_weights)[:self.LOCAL_FIX_VAR] + list(map
                (lambda p: p[1] - self.update_lr * p[0], zip(grad[self.
                LOCAL_FIX_VAR:], fast_weights[self.LOCAL_FIX_VAR:])))
        del net
        return fast_weights

    def finetuning_predict(self, x_qry, y_qry, fast_weights, poiid_emb,
        scaler=None, meta_feature=None):
        x_uid_qry, x_hist_qry, x_candi_qry = x_qry
        y_pred = []
        y_pred_prob = []
        logits_q = self.net(x_uid_qry, x_hist_qry, x_candi_qry,
            fast_weights, scaler=scaler)
        with paddle.no_grad():
            pred_q = F.softmax(logits_q, axis=-1).argmax(axis=-1)
            y_pred.extend(pred_q.data.detach().cpu().numpy().tolist())
            # y_pred_prob.extend(logits_q.softmax(dim=-1)[:, 1].data.detach()
            #     .cpu().numpy().tolist())
            y_pred_prob.extend(F.softmax(logits_q, axis=-1)[:, 1].numpy().tolist())
        return y_pred, y_pred_prob
