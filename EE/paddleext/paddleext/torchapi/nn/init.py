"""
init function for paddle
"""
import paddle


def normal_(tensor, mean=0.0, std=1.0):
    """

    Args:
        tensor:
        mean:
        std:

    Returns:

    """

    paddle.assign(paddle.normal(mean=mean, std=std, shape=tensor.shape), tensor)

    return tensor

def zeros_(tensor):
    """

    Args:
        tensor:

    Returns:

    """

    paddle.assign(paddle.zeros_like(tensor), tensor)

    return tensor

def ones_(tensor):
    """

    Args:
        tensor:

    Returns:

    """

    paddle.assign(paddle.ones_like(tensor), tensor)

    return tensor