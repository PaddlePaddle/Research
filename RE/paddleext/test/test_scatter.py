from chaos.backend_ import paddle_ as paddle, torch_ as torch
import numpy as np

def test_scatter_1d():

    x = np.random.rand(100)

    indices = np.random.randint(low=0, high=100, size=50)
    updates = np.random.rand(50)

    paddle_out = paddle.scatter(paddle.from_numpy(x), 0, paddle.from_numpy(indices), paddle.from_numpy(updates))
    torch_out = torch.scatter(torch.from_numpy(x), 0, torch.from_numpy(indices), torch.from_numpy(updates))

    paddle_out = paddle_out.numpy()
    torch_out = torch_out.numpy()

    assert np.all(paddle_out == torch_out)


def test_scatter_2d_dim0():

    dim0 = 101
    dim1 = 31
    x = np.random.rand(dim0, dim1)

    # for dim = 0

    import random

    indices = list(range(dim0))
    random.shuffle(indices)
    indices = np.array(indices[:50]).reshape((25, 2))
    updates = np.random.rand(indices.shape[0], 2)

    torch_out = torch.scatter(torch.from_numpy(x), 0, torch.from_numpy(indices), torch.from_numpy(updates))
    paddle_out = paddle.scatter(paddle.from_numpy(x), 0, paddle.from_numpy(indices), paddle.from_numpy(updates))

    paddle_out = paddle_out.numpy()
    torch_out = torch_out.numpy()

    assert np.allclose(paddle_out, torch_out)


def test_scatter_2d_dim1():

    dim0 = 101
    dim1 = 131
    x = np.random.rand(dim0, dim1)

    # for dim = 0

    import random

    indices = list(range(dim1))
    random.shuffle(indices)
    indices = np.array(indices[:50]).reshape((25, 2))
    updates = np.random.rand(indices.shape[0], 2)

    torch_out = torch.scatter(torch.from_numpy(x), 1, torch.from_numpy(indices), torch.from_numpy(updates))
    paddle_out = paddle.scatter(paddle.from_numpy(x), 1, paddle.from_numpy(indices), paddle.from_numpy(updates))

    paddle_out = paddle_out.numpy()
    torch_out = torch_out.numpy()

    assert np.allclose(paddle_out, torch_out)


def test_scatter_nd_dimm():
    import random, math

    for rank in range(1, 6):
        for test in range(10):
            dim = random.randint(0, rank-1)

            shape = [random.randint(5, 10) for _ in range(rank)]

            indice_shape = [random.randint(5, 10) for _ in range(rank)]
            indice_shape = [min(shape[i], indice_shape[i]) for i in range(rank)]
            indice_numel = math.prod(indice_shape)

            shape[dim] = 2 * indice_numel

            x = np.random.rand(*shape)

            indice_value = list(range(shape[dim]))
            random.shuffle(indice_value)

            indices = np.array(indice_value[:indice_numel]).reshape(indice_shape)
            updates = np.random.rand(*indice_shape)

            torch_out = torch.scatter(torch.from_numpy(x), dim, torch.from_numpy(indices), torch.from_numpy(updates))
            paddle_out = paddle.scatter(paddle.from_numpy(x), dim, paddle.from_numpy(indices), paddle.from_numpy(updates))

            paddle_out = paddle_out.numpy()
            torch_out = torch_out.numpy()

            assert np.allclose(paddle_out, torch_out)

def test_scatter_add_1d():

    x = np.random.rand(100)

    indices = np.random.randint(low=0, high=100, size=50)
    updates = np.random.rand(50)

    paddle_out = paddle.scatter_add(paddle.from_numpy(x), 0, paddle.from_numpy(indices), paddle.from_numpy(updates))
    torch_out = torch.scatter_add(torch.from_numpy(x), 0, torch.from_numpy(indices), torch.from_numpy(updates))

    paddle_out = paddle_out.numpy()
    torch_out = torch_out.numpy()

    assert np.all(paddle_out == torch_out)

def test_scatter_add_nd_dimm():
    import random, math

    for rank in range(1, 6):
        for test in range(10):
            dim = random.randint(0, rank-1)

            shape = [random.randint(5, 10) for _ in range(rank)]

            indice_shape = [random.randint(5, 10) for _ in range(rank)]
            indice_shape = [min(shape[i], indice_shape[i]) for i in range(rank)]
            indice_numel = math.prod(indice_shape)

            shape[dim] = 2 * indice_numel

            x = np.random.rand(*shape)


            indice_value = list(range(shape[dim]))
            random.shuffle(indice_value)

            indices = np.array(indice_value[:indice_numel]).reshape(indice_shape)

            # indices = np.random.randint(0, shape[dim], size=indice_shape)
            updates = np.random.rand(*indice_shape)

            torch_out = torch.scatter_add(torch.from_numpy(x), dim, torch.from_numpy(indices), torch.from_numpy(updates))
            paddle_out = paddle.scatter_add(paddle.from_numpy(x), dim, paddle.from_numpy(indices), paddle.from_numpy(updates))

            paddle_out = paddle_out.numpy()
            torch_out = torch_out.numpy()

            assert np.allclose(paddle_out, torch_out)