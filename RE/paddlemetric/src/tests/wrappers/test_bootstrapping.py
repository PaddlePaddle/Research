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
import operator
from functools import partial

import numpy as np
import pytest
import paddleext.torchapi as B
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from paddleext.torchapi import Tensor

from tests.helpers import seed_all
from paddlemetrics import MeanSquaredError, Precision, Recall
from paddlemetrics.utilities import apply_to_collection
from paddlemetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_7
from paddlemetrics.wrappers.bootstrapping import BootStrapper, _bootstrap_sampler

seed_all(42)

_preds = B.randint(10, (10, 32))
_target = B.randint(10, (10, 32))


class TestBootStrapper(BootStrapper):
    """For testing purpose, we subclass the bootstrapper class so we can get the exact permutation the class is
    creating."""

    def update(self, *args) -> None:
        self.out = []
        for idx in range(self.num_bootstraps):
            size = len(args[0])
            sample_idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
            new_args = apply_to_collection(args, Tensor, B.index_select, dim=0, index=sample_idx)
            self.metrics[idx].update(*new_args)
            self.out.append(new_args)


def _sample_checker(old_samples, new_samples, op: operator, threshold: int):
    found_one = False
    for os in old_samples:
        cond = op(os, new_samples)
        if cond.sum() > threshold:
            found_one = True
            break
    return found_one


@pytest.mark.parametrize("sampling_strategy", ["poisson", "multinomial"])
def test_bootstrap_sampler(sampling_strategy):
    """make sure that the bootstrap sampler works as intended."""
    old_samples = B.randn(20, 2)

    # make sure that the new samples are only made up of old samples
    idx = _bootstrap_sampler(20, sampling_strategy=sampling_strategy)
    new_samples = old_samples[idx]
    for ns in new_samples:
        assert ns in old_samples

    found_one = _sample_checker(old_samples, new_samples, operator.eq, 2)
    assert found_one, "resampling did not work because no samples were sampled twice"

    found_zero = _sample_checker(old_samples, new_samples, operator.ne, 0)
    assert found_zero, "resampling did not work because all samples were atleast sampled once"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampling_strategy", ["poisson", "multinomial"])
@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        [Precision(average="micro"), partial(precision_score, average="micro")],
        [Recall(average="micro"), partial(recall_score, average="micro")],
        [MeanSquaredError(), mean_squared_error],
    ],
)
def test_bootstrap(device, sampling_strategy, metric, sk_metric):
    """Test that the different bootstraps gets updated as we expected and that the compute method works."""
    if device == "cuda" and not B.cuda.is_available():
        pytest.skip("Test with device='cuda' requires gpu")

    _kwargs = {"base_metric": metric, "mean": True, "std": True, "raw": True, "sampling_strategy": sampling_strategy}
    if _TORCH_GREATER_EQUAL_1_7:
        _kwargs.update(dict(quantile=B.tensor([0.05, 0.95], device=device)))

    bootstrapper = TestBootStrapper(**_kwargs)
    bootstrapper.to(device)

    collected_preds = [[] for _ in range(10)]
    collected_target = [[] for _ in range(10)]
    for p, t in zip(_preds, _target):
        p, t = p.to(device), t.to(device)
        bootstrapper.update(p, t)

        for i, o in enumerate(bootstrapper.out):

            collected_preds[i].append(o[0])
            collected_target[i].append(o[1])

    collected_preds = [B.cat(cp).cpu() for cp in collected_preds]
    collected_target = [B.cat(ct).cpu() for ct in collected_target]

    sk_scores = [sk_metric(ct, cp) for ct, cp in zip(collected_target, collected_preds)]

    output = bootstrapper.compute()
    # quantile only avaible for pytorch v1.7 and forward
    if _TORCH_GREATER_EQUAL_1_7:
        assert np.allclose(output["quantile"][0].cpu(), np.quantile(sk_scores, 0.05))
        assert np.allclose(output["quantile"][1].cpu(), np.quantile(sk_scores, 0.95))

    assert np.allclose(output["mean"].cpu(), np.mean(sk_scores))
    assert np.allclose(output["std"].cpu(), np.std(sk_scores, ddof=1))
    assert np.allclose(output["raw"].cpu(), sk_scores)
