
import numpy as np

from chaos.backend_ import paddle_ as paddle, torch_ as torch


def test_pad():
    import random

    for ndim in range(2, 6):
        for test in range(5):
            shape = [random.randint(5, 10) for _ in range(ndim)]
            x = np.random.rand(*shape)

            torch_input = torch.from_numpy(x)
            paddle_input = paddle.from_numpy(x)

            for rank in range(1, ndim + 1):

                pad = [random.randint(0, 10) for _ in range(rank)] + [random.randint(0, 10) for _ in range(rank)]

                torch_output = torch.nn.functional.pad(torch_input, pad, mode='constant', value=0.0)

                paddle_output = paddle.nn.functional.pad(paddle_input, pad, mode='constant', value=0.0)

                paddle_out = paddle_output.numpy()
                torch_out = torch_output.numpy()

                assert np.allclose(paddle_out, torch_out)