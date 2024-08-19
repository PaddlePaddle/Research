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
from collections import namedtuple
from functools import partial

import pytest
import paddleext.torchapi as B
from skimage.metrics import structural_similarity

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from paddlemetrics.functional import ssim
from paddlemetrics.image import SSIM

seed_all(42)

Input = namedtuple("Input", ["preds", "target", "multichannel"])

_inputs = []
for size, channel, coef, multichannel, dtype in [
    (12, 3, 0.9, True, B.float),
    (13, 1, 0.8, False, B.float32),
    (14, 1, 0.7, False, B.double),
    (15, 3, 0.6, True, B.float64),
]:
    preds = B.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        Input(
            preds=preds,
            target=preds * coef,
            multichannel=multichannel,
        )
    )


def _sk_ssim(preds, target, data_range, multichannel, kernel_size):
    c, h, w = preds.shape[-3:]
    sk_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    sk_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    if not multichannel:
        sk_preds = sk_preds[:, :, :, 0]
        sk_target = sk_target[:, :, :, 0]

    return structural_similarity(
        sk_target,
        sk_preds,
        data_range=data_range,
        multichannel=multichannel,
        gaussian_weights=True,
        win_size=kernel_size,
        sigma=1.5,
        use_sample_covariance=False,
    )


@pytest.mark.parametrize(
    "preds, target, multichannel",
    [(i.preds, i.target, i.multichannel) for i in _inputs],
)
@pytest.mark.parametrize("kernel_size", [5, 11])
class TestSSIM(MetricTester):
    atol = 6e-3

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_ssim(self, preds, target, multichannel, kernel_size, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SSIM,
            partial(_sk_ssim, data_range=1.0, multichannel=multichannel, kernel_size=kernel_size),
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size)},
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_ssim_functional(self, preds, target, multichannel, kernel_size):
        self.run_functional_metric_test(
            preds,
            target,
            ssim,
            partial(_sk_ssim, data_range=1.0, multichannel=multichannel, kernel_size=kernel_size),
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size)},
        )

    # SSIM half + cpu does not work due to missing support in B.log
    @pytest.mark.xfail(reason="SSIM metric does not support cpu + half precision")
    def test_ssim_half_cpu(self, preds, target, multichannel, kernel_size):
        self.run_precision_test_cpu(preds, target, SSIM, ssim, {"data_range": 1.0})

    @pytest.mark.skipif(not B.cuda.is_available(), reason="test requires cuda")
    def test_ssim_half_gpu(self, preds, target, multichannel, kernel_size):
        self.run_precision_test_gpu(preds, target, SSIM, ssim, {"data_range": 1.0})


@pytest.mark.parametrize(
    ["pred", "target", "kernel", "sigma"],
    [
        pytest.param([1, 16, 16], [1, 16, 16], [11, 11], [1.5, 1.5]),  # len(shape)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5, 1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 10], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, -11], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5, 0]),  # invalid sigma input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, -1.5]),  # invalid sigma input
    ],
)
def test_ssim_invalid_inputs(pred, target, kernel, sigma):
    pred_t = B.rand(pred)
    target_t = B.rand(target, dtype=B.float64)
    with pytest.raises(TypeError):
        ssim(pred_t, target_t)

    pred = B.rand(pred)
    target = B.rand(target)
    with pytest.raises(ValueError):
        ssim(pred, target, kernel, sigma)


def test_ssim_unequal_kernel_size():
    """Test the case where kernel_size[0] != kernel_size[1]"""
    preds = B.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    target = B.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ]
        ]
    )
    # kernel order matters
    assert ssim(preds, target, kernel_size=(3, 5)) == B.tensor(0.10814697)
    assert ssim(preds, target, kernel_size=(5, 3)) != B.tensor(0.10814697)
