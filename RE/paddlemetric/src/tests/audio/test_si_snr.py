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
import speechmetrics
import paddleext.torchapi as B
from paddleext.torchapi import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from paddlemetrics.audio import SI_SNR
from paddlemetrics.functional import si_snr
from paddlemetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Time = 100

Input = namedtuple("Input", ["preds", "target"])

inputs = Input(
    preds=B.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
    target=B.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
)

speechmetrics_sisdr = speechmetrics.load("sisdr")


def speechmetrics_si_sdr(preds: Tensor, target: Tensor, zero_mean: bool = True):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    if zero_mean:
        preds = preds - preds.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            metric = speechmetrics_sisdr(preds[i, j], target[i, j], rate=16000)
            ms.append(metric["sisdr"][0])
        mss.append(ms)
    return B.tensor(mss)


def average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (inputs.preds, inputs.target, speechmetrics_si_sdr),
    ],
)
class TestSISNR(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_si_snr(self, preds, target, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SI_SNR,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_si_snr_functional(self, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            si_snr,
            sk_metric,
        )

    def test_si_snr_differentiability(self, preds, target, sk_metric):
        self.run_differentiability_test(preds=preds, target=target, metric_module=SI_SNR, metric_functional=si_snr)

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_si_snr_half_cpu(self, preds, target, sk_metric):
        pytest.xfail("SI-SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not B.cuda.is_available(), reason="test requires cuda")
    def test_si_snr_half_gpu(self, preds, target, sk_metric):
        self.run_precision_test_gpu(preds=preds, target=target, metric_module=SI_SNR, metric_functional=si_snr)


def test_error_on_different_shape(metric_class=SI_SNR):
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(B.randn(100), B.randn(50))
