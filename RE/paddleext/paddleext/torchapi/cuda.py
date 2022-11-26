
import paddle

_initialized=True
def is_available():

    return paddle.device.cuda.device_count() > 0

def manual_seed_all(seed):
    paddle.seed(seed)


def manual_seed(seed):
    paddle.seed(seed)


def set_device(device):
    return paddle.set_device(device)


def empty_cache():
    return


def device_count():
    
    return paddle.device.cuda.device_count()