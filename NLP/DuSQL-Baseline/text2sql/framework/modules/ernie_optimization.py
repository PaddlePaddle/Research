# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
ernie optimization
"""

import collections
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet


def append_cast_op(i, o, prog):
    """
    Append a cast op in a given Program to cast input `i` to data type `o.dtype`.
    Args:
        i (Variable): The input Variable.
        o (Variable): The output Variable.
        prog (Program): The Program to append cast op.
    """
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={"in_dtype": i.dtype,
               "out_dtype": o.dtype})


def copy_to_master_param(p, block):
    """ copy_to_master_param """
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def apply_dynamic_loss_scaling(loss_scaling, master_params_grads,
                               incr_every_n_steps, decr_every_n_nan_or_inf,
                               incr_ratio, decr_ratio):
    """ apply_dynamic_loss_scaling """
    _incr_every_n_steps = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=incr_every_n_steps)
    _decr_every_n_nan_or_inf = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)

    _num_good_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_good_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)
    _num_bad_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_bad_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)

    grads = [fluid.layers.reduce_sum(g) for [_, g] in master_params_grads]
    all_grads = fluid.layers.concat(grads)
    all_grads_sum = fluid.layers.reduce_sum(all_grads)
    is_overall_finite = fluid.layers.isfinite(all_grads_sum)

    update_loss_scaling(is_overall_finite, loss_scaling, _num_good_steps,
                        _num_bad_steps, _incr_every_n_steps,
                        _decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

    # apply_gradient append all ops in global block, thus we shouldn't
    # apply gradient in the switch branch.
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            pass
        with switch.default():
            for _, g in master_params_grads:
                fluid.layers.assign(fluid.layers.zeros_like(g), g)


def create_master_params_grads(params_grads, main_prog, startup_prog,
                               loss_scaling):
    """ create_master_params_grads """
    master_params_grads = []
    with main_prog._backward_role_guard():
        for p, g in params_grads:
            # create master parameters
            master_param = copy_to_master_param(p, main_prog.global_block())
            startup_master_param = startup_prog.global_block()._clone_variable(
                master_param)
            startup_p = startup_prog.global_block().var(p.name)
            append_cast_op(startup_p, startup_master_param, startup_prog)
            # cast fp16 gradients to fp32 before apply gradients
            if g.name.find("layer_norm") > -1:
                scaled_g = g / loss_scaling
                master_params_grads.append([p, scaled_g])
                continue
            master_grad = fluid.layers.cast(g, "float32")
            master_grad = master_grad / loss_scaling
            master_params_grads.append([master_param, master_grad])

    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    """ master_param_to_train_param """
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            if train_p.name.find("layer_norm") > -1:
                fluid.layers.assign(m_p_g[0], train_p)
            else:
                append_cast_op(m_p_g[0], train_p, main_prog)


def update_loss_scaling(is_overall_finite, prev_loss_scaling, num_good_steps,
                        num_bad_steps, incr_every_n_steps,
                        decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
    """
    Update loss scaling according to overall gradients. If all gradients is
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio.
    Otherwisw, loss scaling will decrease by decr_ratio after
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.
    Args:
        is_overall_finite (Variable): A boolean variable indicates whether
                                     all gradients are finite.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which
                                  some gradients are infinite.
        incr_every_n_steps (Variable): A variable represents increasing loss
                                       scaling every n consecutive steps with
                                       finite gradients.
        decr_every_n_nan_or_inf (Variable): A variable represents decreasing
                                            loss scaling every n accumulated
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing
                           loss scaling.
    """
    zero_steps = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            should_incr_loss_scaling = fluid.layers.less_than(
                incr_every_n_steps, num_good_steps + 1)
            with fluid.layers.Switch() as switch1:
                with switch1.case(should_incr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * incr_ratio
                    loss_scaling_is_finite = fluid.layers.isfinite(
                        new_loss_scaling)
                    with fluid.layers.Switch() as switch2:
                        with switch2.case(loss_scaling_is_finite):
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                        with switch2.default():
                            pass
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

                with switch1.default():
                    fluid.layers.increment(num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

        with switch.default():
            should_decr_loss_scaling = fluid.layers.less_than(
                decr_every_n_nan_or_inf, num_bad_steps + 1)
            with fluid.layers.Switch() as switch3:
                with switch3.case(should_decr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * decr_ratio
                    static_loss_scaling = \
                        fluid.layers.fill_constant(shape=[1],
                                                   dtype='float32',
                                                   value=1.0)
                    less_than_one = fluid.layers.less_than(new_loss_scaling,
                                                           static_loss_scaling)
                    with fluid.layers.Switch() as switch4:
                        with switch4.case(less_than_one):
                            fluid.layers.assign(static_loss_scaling,
                                                prev_loss_scaling)
                        with switch4.default():
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)
                with switch3.default():
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.increment(num_bad_steps)


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
        )

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 use_dynamic_loss_scaling=False,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8,
                 dist_strategy=None):
    """ optimization """
    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler \
                .noam_decay(1 / (warmup_steps * (learning_rate ** 2)),
                            warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr, epsilon=1e-06)
        # optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    else:
        scheduled_lr = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("learning_rate"),
            shape=[1],
            value=learning_rate,
            dtype='float32',
            persistable=True)
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr, epsilon=1e-06)
        # optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
        optimizer._learning_rate_map[fluid.default_main_program(
        )] = scheduled_lr

    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    def exclude_from_weight_decay(name):
        """ exclude_from_weight_decay """
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    loss_scaling = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("loss_scaling"),
        shape=[1],
        value=init_loss_scaling,
        dtype='float32',
        persistable=True)

    if use_fp16:
        loss *= loss_scaling
        param_grads = optimizer.backward(loss)

        master_param_grads = create_master_params_grads(
            param_grads, train_program, startup_prog, loss_scaling)

        for param, _ in master_param_grads:
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        if use_dynamic_loss_scaling:
            apply_dynamic_loss_scaling(
                loss_scaling, master_param_grads, incr_every_n_steps,
                decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

        optimizer.apply_gradients(master_param_grads)

        if weight_decay > 0:
            for param, grad in master_param_grads:
                if exclude_from_weight_decay(param.name.rstrip(".master")):
                    continue
                with param.block.program._optimized_guard(
                        [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

        master_param_to_train_param(master_param_grads, param_grads,
                                    train_program)

    else:
        for param in train_program.global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        if dist_strategy is not None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

        _, param_grads = optimizer.minimize(loss)

        if weight_decay > 0:
            for param, grad in param_grads:
                if exclude_from_weight_decay(param.name):
                    continue
                with param.block.program._optimized_guard(
                        [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)
    result = collections.OrderedDict()
    result['scheduled_lr'] = scheduled_lr
    if use_fp16:
        result['loss_scaling'] = loss_scaling
    return result
