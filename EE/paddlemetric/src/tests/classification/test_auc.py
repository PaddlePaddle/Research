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

import numpy as np
import pytest
from sklearn.metrics import auc as _sk_auc
from paddleext.torchapi import tensor

from tests.helpers import seed_all
from tests.helpers.testers import NUM_BATCHES, MetricTester
from paddlemetrics.classification.auc import AUC
from paddlemetrics.functional import auc

seed_all(42)


def sk_auc(x, y, reorder=False):
    x = x.flatten()
    y = y.flatten()
    if reorder:
        idx = np.argsort(x, kind="stable")
        x = x[idx]
        y = y[idx]
    return _sk_auc(x, y)


Input = namedtuple("Input", ["x", "y"])

_examples = []
# generate already ordered samples, sorted in both directions
for batch_size in (8, 4049):
    for i in range(4):
        x = np.random.rand(NUM_BATCHES * batch_size)
        y = np.random.rand(NUM_BATCHES * batch_size)
        idx = np.argsort(x, kind="stable")
        x = x[idx] if i % 2 == 0 else x[idx[::-1]]
        y = y[idx] if i % 2 == 0 else x[idx[::-1]]
        x = x.reshape(NUM_BATCHES, batch_size)
        y = y.reshape(NUM_BATCHES, batch_size)
        _examples.append(Input(x=tensor(x), y=tensor(y)))


@pytest.mark.parametrize("x, y", _examples)
class TestAUC(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_auc(self, x, y, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=x,
            target=y,
            metric_class=AUC,
            sk_metric=sk_auc,
            dist_sync_on_step=dist_sync_on_step,
        )

    @pytest.mark.parametrize("reorder", [True, False])
    def test_auc_functional(self, x, y, reorder):
        self.run_functional_metric_test(
            x, y, metric_functional=auc, sk_metric=partial(sk_auc, reorder=reorder), metric_args={"reorder": reorder}
        )

    @pytest.mark.parametrize("reorder", [True, False])
    def test_auc_differentiability(self, x, y, reorder):
        self.run_differentiability_test(
            preds=x, target=y, metric_module=AUC, metric_functional=auc, metric_args={"reorder": reorder}
        )


@pytest.mark.parametrize("unsqueeze_x", (True, False))
@pytest.mark.parametrize("unsqueeze_y", (True, False))
@pytest.mark.parametrize(
    ["x", "y", "expected"],
    [
        pytest.param([0, 1], [0, 1], 0.5),
        pytest.param([1, 0], [0, 1], 0.5),
        pytest.param([1, 0, 0], [0, 1, 1], 0.5),
        pytest.param([0, 1], [1, 1], 1),
        pytest.param([0, 0.5, 1], [0, 0.5, 1], 0.5),
    ],
)
def test_auc(x, y, expected, unsqueeze_x, unsqueeze_y):
    x = tensor(x)
    y = tensor(y)

    if unsqueeze_x:
        x = x.unsqueeze(-1)

    if unsqueeze_y:
        y = y.unsqueeze(-1)

    # Test Area Under Curve (AUC) computation
    assert auc(x, y, reorder=True) == expected
