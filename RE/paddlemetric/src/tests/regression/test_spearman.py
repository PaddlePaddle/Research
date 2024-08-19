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

import pytest
import paddleext.torchapi as B
from scipy.stats import rankdata, spearmanr

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from paddlemetrics.functional.regression.spearman import _rank_data, spearman_corrcoef
from paddlemetrics.regression.spearman import SpearmanCorrcoef

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_single_target_inputs1 = Input(
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.rand(NUM_BATCHES, BATCH_SIZE),
)

_single_target_inputs2 = Input(
    preds=B.randn(NUM_BATCHES, BATCH_SIZE),
    target=B.randn(NUM_BATCHES, BATCH_SIZE),
)

_specific_input = Input(
    preds=B.stack([B.tensor([1.0, 0.0, 4.0, 1.0, 0.0, 3.0, 0.0]) for _ in range(NUM_BATCHES)]),
    target=B.stack([B.tensor([4.0, 0.0, 3.0, 3.0, 3.0, 1.0, 1.0]) for _ in range(NUM_BATCHES)]),
)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_specific_input.preds, _specific_input.target),
    ],
)
def test_ranking(preds, target):
    """test that ranking function works as expected."""
    for p, t in zip(preds, target):
        scipy_ranking = [rankdata(p.numpy()), rankdata(t.numpy())]
        tm_ranking = [_rank_data(p), _rank_data(t)]
        assert (B.tensor(scipy_ranking[0]) == tm_ranking[0]).all()
        assert (B.tensor(scipy_ranking[1]) == tm_ranking[1]).all()


def _sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return spearmanr(sk_target, sk_preds)[0]


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_specific_input.preds, _specific_input.target),
    ],
)
class TestSpearmanCorrcoef(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_spearman_corrcoef(self, preds, target, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpearmanCorrcoef,
            _sk_metric,
            dist_sync_on_step,
        )

    def test_spearman_corrcoef_functional(self, preds, target):
        self.run_functional_metric_test(preds, target, spearman_corrcoef, _sk_metric)

    def test_spearman_corrcoef_differentiability(self, preds, target):
        self.run_differentiability_test(
            preds=preds, target=target, metric_module=SpearmanCorrcoef, metric_functional=spearman_corrcoef
        )

    # Spearman half + cpu does not work due to missing support in B.arange
    @pytest.mark.xfail(reason="Spearman metric does not support cpu + half precision")
    def test_spearman_corrcoef_half_cpu(self, preds, target):
        self.run_precision_test_cpu(preds, target, SpearmanCorrcoef, spearman_corrcoef)

    @pytest.mark.skipif(not B.cuda.is_available(), reason="test requires cuda")
    def test_spearman_corrcoef_half_gpu(self, preds, target):
        self.run_precision_test_gpu(preds, target, SpearmanCorrcoef, spearman_corrcoef)


def test_error_on_different_shape():
    metric = SpearmanCorrcoef()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(B.randn(100), B.randn(50))

    with pytest.raises(ValueError, match="Expected both predictions and target to be 1 dimensional tensors."):
        metric(B.randn(100, 2), B.randn(100, 2))
