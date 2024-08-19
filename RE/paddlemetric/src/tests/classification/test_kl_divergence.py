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
from typing import Optional

import numpy as np
import pytest
import paddleext.torchapi as B
from scipy.stats import entropy
from paddleext.torchapi import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, MetricTester
from paddlemetrics.classification import KLDivergence
from paddlemetrics.functional import kl_divergence

seed_all(42)

Input = namedtuple("Input", ["p", "q"])

_probs_inputs = Input(
    p=B.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    q=B.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_log_probs_inputs = Input(
    p=B.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
    q=B.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
)


def _sk_metric(p: Tensor, q: Tensor, log_prob: bool, reduction: Optional[str] = "mean"):
    if log_prob:
        p = p.softmax(dim=-1)
        q = q.softmax(dim=-1)
    res = entropy(p, q, axis=1)
    if reduction == "mean":
        return np.mean(res)
    if reduction == "sum":
        return np.sum(res)
    return res


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "p, q, log_prob", [(_probs_inputs.p, _probs_inputs.q, False), (_log_probs_inputs.p, _log_probs_inputs.q, True)]
)
class TestKLDivergence(MetricTester):
    atol = 1e-6

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_kldivergence(self, reduction, p, q, log_prob, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            p,
            q,
            KLDivergence,
            partial(_sk_metric, log_prob=log_prob, reduction=reduction),
            dist_sync_on_step,
            metric_args=dict(log_prob=log_prob, reduction=reduction),
        )

    def test_kldivergence_functional(self, reduction, p, q, log_prob):
        # todo: `num_outputs` is unused
        self.run_functional_metric_test(
            p,
            q,
            kl_divergence,
            partial(_sk_metric, log_prob=log_prob, reduction=reduction),
            metric_args=dict(log_prob=log_prob, reduction=reduction),
        )

    def test_kldivergence_differentiability(self, reduction, p, q, log_prob):
        self.run_differentiability_test(
            p,
            q,
            metric_module=KLDivergence,
            metric_functional=kl_divergence,
            metric_args=dict(log_prob=log_prob, reduction=reduction),
        )

    # KLDivergence half + cpu does not work due to missing support in B.clamp
    @pytest.mark.xfail(reason="KLDivergence metric does not support cpu + half precision")
    def test_kldivergence_half_cpu(self, reduction, p, q, log_prob):
        self.run_precision_test_cpu(p, q, KLDivergence, kl_divergence, {"log_prob": log_prob, "reduction": reduction})

    @pytest.mark.skipif(not B.cuda.is_available(), reason="test requires cuda")
    def test_r2_half_gpu(self, reduction, p, q, log_prob):
        self.run_precision_test_gpu(p, q, KLDivergence, kl_divergence, {"log_prob": log_prob, "reduction": reduction})


def test_error_on_different_shape():
    metric = KLDivergence()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(B.randn(100), B.randn(50))


def test_error_on_multidim_tensors():
    metric = KLDivergence()
    with pytest.raises(ValueError, match="Expected both p and q distribution to be 2D but got 3 and 3 respectively"):
        metric(B.randn(10, 20, 5), B.randn(10, 20, 5))
