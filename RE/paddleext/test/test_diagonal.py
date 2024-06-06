
import numpy as np

from chaos import paddle_ as paddle, torch_ as torch


def test_diagonal():
    import random

    for rank in range(2, 6):
        for test in range(10):
            while True:
                dim1 = random.randint(0, rank - 1)
                dim2 = random.randint(0, rank - 1)
                if dim1 != dim2:
                    break

            shape = [random.randint(5, 10) for _ in range(rank)]
            offset = random.randint(-shape[dim1] + 1, shape[dim2])

            x = np.random.rand(*shape)

            torch_input = torch.from_numpy(x)
            torch.fill_diagonal(torch_input, value=100, offset=offset, dim1=dim1, dim2=dim2)

            paddle_input = paddle.from_numpy(x)
            paddle.fill_diagonal(paddle_input, value=100, offset=offset, dim1=dim1, dim2=dim2)

            paddle_out = paddle_input.numpy()
            torch_out = torch_input.numpy()

            assert np.sum(np.abs(paddle_out - torch_out)) < 1e-5