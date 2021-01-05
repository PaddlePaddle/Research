# -*- coding: utf-8
import os
import paddle.fluid as fluid
# 调用load_op_library加载动态库
file_dir = os.path.dirname(os.path.abspath(__file__))  
fluid.load_op_library(os.path.join(file_dir, 'lib/rerank.so'))

from paddle.fluid.layer_helper import LayerHelper

def vmat(x, name=None):
    # vmat的type和在OP中定义的type相同
    helper = LayerHelper("vmat", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="vmat", inputs={"X": x}, outputs={"Y": out})
    return out


def qe(V,R,S, name=None):
    # qe的type和在OP中定义的type相同
    helper = LayerHelper("qe", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=V.dtype)
    helper.append_op(type="qe", inputs={"V": V,"R":R, "S":S}, outputs={"Y": out})
    return out
