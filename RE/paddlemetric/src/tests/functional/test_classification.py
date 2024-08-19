# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import paddleext.torchapi as B
from paddleext.torchapi import Tensor, tensor

from tests.helpers import seed_all
from paddlemetrics.functional import dice_score
from paddlemetrics.functional.classification.precision_recall_curve import _binary_clf_curve
from paddlemetrics.utilities.data import get_num_classes, to_categorical, to_onehot


def test_onehot():
    test_tensor = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = B.stack(
        [
            B.cat([B.eye(5, dtype=int), B.zeros((5, 5), dtype=int)]),
            B.cat([B.zeros((5, 5), dtype=int), B.eye(5, dtype=int)]),
        ]
    )

    assert tuple(test_tensor.shape) == (2, 5)
    assert tuple(expected.shape) == (2, 10, 5)

    onehot_classes = to_onehot(test_tensor, num_classes=10)
    onehot_no_classes = to_onehot(test_tensor)

    assert B.allclose(onehot_classes, onehot_no_classes)

    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape

    assert B.allclose(expected.to(onehot_no_classes), onehot_no_classes)
    assert B.allclose(expected.to(onehot_classes), onehot_classes)


def test_to_categorical():
    test_tensor = B.stack(
        [
            B.cat([B.eye(5, dtype=int), B.zeros((5, 5), dtype=int)]),
            B.cat([B.zeros((5, 5), dtype=int), B.eye(5, dtype=int)]),
        ]
    ).to(B.float)

    expected = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert tuple(expected.shape) == (2, 5)
    assert tuple(test_tensor.shape) == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert B.allclose(result, expected.to(result.dtype))


@pytest.mark.parametrize(
    ["preds", "target", "num_classes", "expected_num_classes"],
    [
        pytest.param(B.rand(32, 10, 28, 28), B.randint(10, (32, 28, 28)), 10, 10),
        pytest.param(B.rand(32, 10, 28, 28), B.randint(10, (32, 28, 28)), None, 10),
        pytest.param(B.rand(32, 28, 28), B.randint(10, (32, 28, 28)), None, 10),
    ],
)
def test_get_num_classes(preds, target, num_classes, expected_num_classes):
    assert get_num_classes(preds, target, num_classes) == expected_num_classes


@pytest.mark.parametrize(
    ["sample_weight", "pos_label", "exp_shape"],
    [
        pytest.param(1, 1.0, 42),
        pytest.param(None, 1.0, 42),
    ],
)
def test_binary_clf_curve(sample_weight, pos_label, exp_shape):
    # TODO: move back the pred and target to test func arguments
    #  if you fix the array inside the function, you'd also have fix the shape,
    #  because when the array changes, you also have to fix the shape
    seed_all(0)
    pred = B.randint(low=51, high=99, size=(100,), dtype=B.float) / 100
    target = tensor([0, 1] * 50, dtype=B.int)
    if sample_weight is not None:
        sample_weight = B.ones_like(pred) * sample_weight

    fps, tps, thresh = _binary_clf_curve(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)

    assert isinstance(tps, Tensor)
    assert isinstance(fps, Tensor)
    assert isinstance(thresh, Tensor)
    if B.platform() == "torch":
        assert tuple(tps.shape) == (exp_shape,)
        assert tuple(fps.shape) == (exp_shape,)
        assert tuple(thresh.shape) == (exp_shape,)
    elif B.platform() == "paddle":
        assert tuple(tps.shape) == (exp_shape - 1,)
        assert tuple(fps.shape) == (exp_shape - 1,)
        assert tuple(thresh.shape) == (exp_shape - 1,)
    else:
        raise Exception(f"unknown platform {B.platform()}")


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        pytest.param([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.0),
        pytest.param([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.0),
        pytest.param([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
        pytest.param([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.0),
    ],
)
def test_dice_score(pred, target, expected):
    score = dice_score(tensor(pred), tensor(target))
    assert score == expected
