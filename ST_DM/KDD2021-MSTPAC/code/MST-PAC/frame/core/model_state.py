"""
Tools for manipulating sets of variables.
"""

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
import os

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    res = []
    for variables in zip(*var_seqs):
        res.append(np.mean(variables, axis=0))
    return res

def subtract_vars(var_seq_1, var_seq_2):
    """
    Subtract one variable sequence from another.
    """
    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]
    

def add_vars(var_seq_1, var_seq_2):
    """
    Add two variable sequences.
    """
    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
    """
    Scale a variable sequence.
    """
    return [v * scale for v in var_seq]

#def weight_decay(rate, variables=None):
#    """
#    Create an Op that performs weight decay.
#    """
#    if variables is None:
#        variables = tf.trainable_variables()
#    ops = [tf.assign(var, var * rate) for var in variables]
#    return tf.group(*ops)

class VariableState:
    """
    Manage the state of a set of variables.
    """
    def __init__(self, prog, variables):
        self._prog = prog
        self._variables = variables # var names

    def export_variables(self):
        """
        Save the current variables.
        """
        return [np.array(fluid.global_scope().var(name).get_tensor()) for name in self._variables] 

    def import_variables(self, values):
        """
        Restore the variables.
        """
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        #print('current gpu_id:', str(gpu_id))
        [fluid.global_scope().var(name).get_tensor().set(value, fluid.CUDAPlace(gpu_id)) for name, value in zip(self._variables, values)]
    
    def broadcast_vars(self, exe):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            fetch_list = []
            for name in self._variables:
                tensor = fluid.global_scope().var(name)
                broadcast_var = fluid.layers.collective._c_broadcast(tensor, root=0, use_calc_stream=True)
                #broadcast_var = fluid.layers.collective._c_broadcast(name, root=0, use_calc_stream=True)
                fetch_list.append(broadcast_var)
        exe.run(prog, fetch_list=fetch_list)
            

def dist_get_reduced_vars(exe, var_seqs):
    '''
    Gather a list of vars from all worker by collective operators.
    '''
    prog = fluid.Program()
    with fluid.program_guard(prog):
        i = 0
        feed_dict = {}
        fetch_list = []
        for var in var_seqs:
            placeholder_name = 'reduce_value_' + str(i)
            var_placeholder = fluid.layers.data(name=placeholder_name, shape=var.shape, dtype='float32')
            reduced_var = fluid.layers.collective._c_allreduce(var_placeholder, reduce_type='sum', use_calc_stream=True) 
            feed_dict[placeholder_name] = var
            fetch_list.append(reduced_var)
            i += 1
    return  exe.run(prog, feed=feed_dict, fetch_list=fetch_list) 