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
from functools import partial

import numpy as np
import pytest
import paddleext.torchapi as B
from sklearn.metrics import jaccard_score as sk_jaccard_score
from paddleext.torchapi import Tensor, tensor

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from paddlemetrics.classification.iou import IoU
from paddlemetrics.functional import iou


def _sk_iou_binary_prob(preds, target, average=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_binary(preds, target, average=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multilabel_prob(preds, target, average=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multilabel(preds, target, average=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multiclass_prob(preds, target, average=None):
    sk_preds = B.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multiclass(preds, target, average=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multidim_multiclass_prob(preds, target, average=None):
    sk_preds = B.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


def _sk_iou_multidim_multiclass(preds, target, average=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_jaccard_score(y_true=sk_target, y_pred=sk_preds, average=average)


@pytest.mark.parametrize("reduction", ["elementwise_mean", "none"])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_iou_binary_prob, 2),
        (_input_binary.preds, _input_binary.target, _sk_iou_binary, 2),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_iou_multilabel_prob, 2),
        (_input_mlb.preds, _input_mlb.target, _sk_iou_multilabel, 2),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_iou_multiclass_prob, NUM_CLASSES),
        (_input_mcls.preds, _input_mcls.target, _sk_iou_multiclass, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_iou_multidim_multiclass_prob, NUM_CLASSES),
        (_input_mdmc.preds, _input_mdmc.target, _sk_iou_multidim_multiclass, NUM_CLASSES),
    ],
)
class TestIoU(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_iou(self, reduction, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        average = "macro" if reduction == "elementwise_mean" else None  # convert tags
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=IoU,
            sk_metric=partial(sk_metric, average=average),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "reduction": reduction},
        )

    def test_iou_functional(self, reduction, preds, target, sk_metric, num_classes):
        average = "macro" if reduction == "elementwise_mean" else None  # convert tags
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=iou,
            sk_metric=partial(sk_metric, average=average),
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "reduction": reduction},
        )

    def test_iou_differentiability(self, reduction, preds, target, sk_metric, num_classes):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=IoU,
            metric_functional=iou,
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "reduction": reduction},
        )


@pytest.mark.parametrize(
    ["half_ones", "reduction", "ignore_index", "expected"],
    [
        pytest.param(False, "none", None, Tensor([1, 1, 1])),
        pytest.param(False, "elementwise_mean", None, Tensor([1])),
        pytest.param(False, "none", 0, Tensor([1, 1])),
        pytest.param(True, "none", None, Tensor([0.5, 0.5, 0.5])),
        pytest.param(True, "elementwise_mean", None, Tensor([0.5])),
        pytest.param(True, "none", 0, Tensor([2 / 3, 1 / 2])),
    ],
)
def test_iou(half_ones, reduction, ignore_index, expected):
    preds = (B.arange(120) % 3).view(-1, 1)
    target = (B.arange(120) % 3).view(-1, 1)
    if half_ones:
        preds[:60] = 1
    iou_val = iou(
        preds=preds,
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    assert B.allclose(iou_val, expected, atol=1e-9)


# test `absent_score`
@pytest.mark.parametrize(
    ["pred", "target", "ignore_index", "absent_score", "num_classes", "expected"],
    [
        # Note that -1 is used as the absent_score in almost all tests here to distinguish it from the range of valid
        # scores the function can return ([0., 1.] range, inclusive).
        # 2 classes, class 0 is correct everywhere, class 1 is absent.
        pytest.param([0], [0], None, -1.0, 2, [1.0, -1.0]),
        pytest.param([0, 0], [0, 0], None, -1.0, 2, [1.0, -1.0]),
        # absent_score not applied if only class 0 is present and it's the only class.
        pytest.param([0], [0], None, -1.0, 1, [1.0]),
        # 2 classes, class 1 is correct everywhere, class 0 is absent.
        pytest.param([1], [1], None, -1.0, 2, [-1.0, 1.0]),
        pytest.param([1, 1], [1, 1], None, -1.0, 2, [-1.0, 1.0]),
        # When 0 index ignored, class 0 does not get a score (not even the absent_score).
        pytest.param([1], [1], 0, -1.0, 2, [1.0]),
        # 3 classes. Only 0 and 2 are present, and are perfectly predicted. 1 should get absent_score.
        pytest.param([0, 2], [0, 2], None, -1.0, 3, [1.0, -1.0, 1.0]),
        pytest.param([2, 0], [2, 0], None, -1.0, 3, [1.0, -1.0, 1.0]),
        # 3 classes. Only 0 and 1 are present, and are perfectly predicted. 2 should get absent_score.
        pytest.param([0, 1], [0, 1], None, -1.0, 3, [1.0, 1.0, -1.0]),
        pytest.param([1, 0], [1, 0], None, -1.0, 3, [1.0, 1.0, -1.0]),
        # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in pred but not target; should not get absent_score), class
        # 2 is absent.
        pytest.param([0, 1], [0, 0], None, -1.0, 3, [0.5, 0.0, -1.0]),
        # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in target but not pred; should not get absent_score), class
        # 2 is absent.
        pytest.param([0, 0], [0, 1], None, -1.0, 3, [0.5, 0.0, -1.0]),
        # Sanity checks with absent_score of 1.0.
        pytest.param([0, 2], [0, 2], None, 1.0, 3, [1.0, 1.0, 1.0]),
        pytest.param([0, 2], [0, 2], 0, 1.0, 3, [1.0, 1.0]),
    ],
)
def test_iou_absent_score(pred, target, ignore_index, absent_score, num_classes, expected):
    iou_val = iou(
        preds=tensor(pred),
        target=tensor(target),
        ignore_index=ignore_index,
        absent_score=absent_score,
        num_classes=num_classes,
        reduction="none",
    )
    assert B.allclose(iou_val, tensor(expected).to(iou_val))


# example data taken from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/tests/test_ranking.py
@pytest.mark.parametrize(
    ["pred", "target", "ignore_index", "num_classes", "reduction", "expected"],
    [
        # Ignoring an index outside of [0, num_classes-1] should have no effect.
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], None, 3, "none", [1, 1 / 2, 2 / 3]),
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], -1, 3, "none", [1, 1 / 2, 2 / 3]),
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 255, 3, "none", [1, 1 / 2, 2 / 3]),
        # Ignoring a valid index drops only that index from the result.
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "none", [1 / 2, 2 / 3]),
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 1, 3, "none", [1, 2 / 3]),
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 2, 3, "none", [1, 1]),
        # When reducing to mean or sum, the ignored index does not contribute to the output.
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "elementwise_mean", [7 / 12]),
        pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, "sum", [7 / 6]),
    ],
)
def test_iou_ignore_index(pred, target, ignore_index, num_classes, reduction, expected):
    iou_val = iou(
        preds=tensor(pred),
        target=tensor(target),
        ignore_index=ignore_index,
        num_classes=num_classes,
        reduction=reduction,
    )
    assert B.allclose(iou_val, tensor(expected).to(iou_val))
