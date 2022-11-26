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
from sklearn.metrics import hamming_loss as sk_hamming_loss

from tests.classification.inputs import _input_binary, _input_binary_logits, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_logits as _input_mcls_logits
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_multidim as _input_mlmd
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import THRESHOLD, MetricTester
from paddlemetrics import HammingDistance
from paddlemetrics.functional import hamming_distance
from paddlemetrics.utilities.checks import _input_format_classification

seed_all(42)


def _sk_hamming_loss(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return sk_hamming_loss(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary_logits.preds, _input_binary_logits.target),
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_binary.preds, _input_binary.target),
        (_input_mlb_logits.preds, _input_mlb_logits.target),
        (_input_mlb_prob.preds, _input_mlb_prob.target),
        (_input_mlb.preds, _input_mlb.target),
        (_input_mcls_logits.preds, _input_mcls_logits.target),
        (_input_mcls_prob.preds, _input_mcls_prob.target),
        (_input_mcls.preds, _input_mcls.target),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target),
        (_input_mdmc.preds, _input_mdmc.target),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target),
        (_input_mlmd.preds, _input_mlmd.target),
    ],
)
class TestHammingDistance(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_hamming_distance_class(self, ddp, dist_sync_on_step, preds, target):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HammingDistance,
            sk_metric=_sk_hamming_loss,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD},
        )

    def test_hamming_distance_fn(self, preds, target):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=hamming_distance,
            sk_metric=_sk_hamming_loss,
            metric_args={"threshold": THRESHOLD},
        )

    def test_hamming_distance_differentiability(self, preds, target):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=HammingDistance,
            metric_functional=hamming_distance,
            metric_args={"threshold": THRESHOLD},
        )


@pytest.mark.parametrize("threshold", [1.5])
def test_wrong_params(threshold):
    preds, target = _input_mcls_prob.preds, _input_mcls_prob.target

    with pytest.raises(ValueError):
        ham_dist = HammingDistance(threshold=threshold)
        ham_dist(preds, target)
        ham_dist.compute()

    with pytest.raises(ValueError):
        hamming_distance(preds, target, threshold=threshold)
