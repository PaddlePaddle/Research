

import paddle
from more_itertools import chunked
from paddle.nn.functional import *

def pad(input, pad, mode='constant', value=0.0):

    pad = sum(reversed(list(chunked(pad, 2))), [])

    if len(pad) < 2 * input.ndim:
        pad = [0] * (2 * input.ndim - len(pad)) + pad

    return paddle.nn.functional.pad(input, pad, mode=mode, value=value)

