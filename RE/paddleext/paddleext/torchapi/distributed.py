
import paddle


def is_available():
    return True

DISTRIBUTED = False

def is_initialized():
    return DISTRIBUTED


def init_process_group(*args, **kwargs):

    pass

