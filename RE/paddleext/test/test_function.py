


from chaos.backend_.paddle_.functional import fill_diagonal
import paddle

def test_fill_diagnonal():

    a = paddle.randn((5, 5))
    fill_diagonal(a, float("-inf"))

if __name__ == "__main__":
    test_fill_diagnonal()