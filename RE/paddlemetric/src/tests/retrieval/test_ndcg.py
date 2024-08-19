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
import numpy as np
import pytest
from sklearn.metrics import ndcg_score
from paddleext.torchapi import Tensor

from tests.helpers import seed_all
from tests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _default_metric_class_input_arguments_with_non_binary_target,
    _default_metric_functional_input_arguments_with_non_binary_target,
    _errors_test_class_metric_parameters_k,
    _errors_test_class_metric_parameters_with_nonbinary,
    _errors_test_functional_metric_parameters_k,
    _errors_test_functional_metric_parameters_with_nonbinary,
)
from paddlemetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from paddlemetrics.retrieval.retrieval_ndcg import RetrievalNormalizedDCG

seed_all(42)


def _ndcg_at_k(target: np.ndarray, preds: np.ndarray, k: int = None):
    """Adapting `from sklearn.metrics.ndcg_score`."""
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if target.shape[0] < 2:  # ranking is equal to ideal ranking with a single document
        return np.array(1.0)

    preds = np.expand_dims(preds, axis=0)
    target = np.expand_dims(target, axis=0)

    return ndcg_score(target, preds, k=k)


class TestNDCG(RetrievalMetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        dist_sync_on_step: bool,
        empty_target_action: str,
        k: int,
    ):
        metric_args = {"empty_target_action": empty_target_action, "k": k}

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalNormalizedDCG,
            sk_metric=_ndcg_at_k,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments_with_non_binary_target)
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    def test_functional_metric(self, preds: Tensor, target: Tensor, k: int):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_normalized_dcg,
            sk_metric=_ndcg_at_k,
            metric_args={},
            k=k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalNormalizedDCG,
            metric_functional=retrieval_normalized_dcg,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalNormalizedDCG,
            metric_functional=retrieval_normalized_dcg,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_with_nonbinary,
            _errors_test_class_metric_parameters_k,
        )
    )
    def test_arguments_class_metric(
        self, indexes: Tensor, preds: Tensor, target: Tensor, message: str, metric_args: dict
    ):
        if target.is_floating_point():
            pytest.skip("NDCG metric works with float target input")

        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalNormalizedDCG,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_functional_metric_parameters_with_nonbinary,
            _errors_test_functional_metric_parameters_k,
        )
    )
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        if target.is_floating_point():
            pytest.skip("NDCG metric works with float target input")

        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_normalized_dcg,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )
