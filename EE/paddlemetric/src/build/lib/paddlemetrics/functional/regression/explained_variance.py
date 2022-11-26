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
from typing import Sequence, Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def _explained_variance_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute Explained Variance. Checks for same shape of input
    tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    _check_same_shape(preds, target)

    n_obs = preds.size(0)
    sum_error = B.sum(target - preds, dim=0)
    diff = target - preds
    sum_squared_error = B.sum(diff * diff, dim=0)

    sum_target = B.sum(target, dim=0)
    sum_squared_target = B.sum(target * target, dim=0)

    return n_obs, sum_error, sum_squared_error, sum_target, sum_squared_target


def _explained_variance_compute(
    n_obs: Tensor,
    sum_error: Tensor,
    sum_squared_error: Tensor,
    sum_target: Tensor,
    sum_squared_target: Tensor,
    multioutput: str = "uniform_average",
) -> Tensor:
    """Computes Explained Variance.

    Args:
        n_obs: Number of predictions or observations
        sum_error: Sum of errors over all observations
        sum_squared_error: Sum of square of errors over all observations
        sum_target: Sum of target values
        sum_squared_target: Sum of squares of target values
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> target = B.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = B.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> n_obs, sum_error, ss_error, sum_target, ss_target = _explained_variance_update(preds, target)
        >>> _explained_variance_compute(n_obs, sum_error, ss_error, sum_target, ss_target, multioutput='raw_values')
        tensor([0.9677, 1.0000])
    """

    diff_avg = sum_error / n_obs
    numerator = sum_squared_error / n_obs - (diff_avg * diff_avg)

    target_avg = sum_target / n_obs
    denominator = sum_squared_target / n_obs - (target_avg * target_avg)

    # Take care of division by zero
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = B.ones_like(diff_avg)
    output_scores[valid_score] = 1.0 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    # Decide what to do in multioutput case
    # Todo: allow user to pass in tensor with weights
    if multioutput == "raw_values":
        return output_scores
    if multioutput == "uniform_average":
        return B.mean(output_scores)
    if multioutput == "variance_weighted":
        denom_sum = B.sum(denominator)
        return B.sum(denominator / denom_sum * output_scores)


def explained_variance(
    preds: Tensor,
    target: Tensor,
    multioutput: str = "uniform_average",
) -> Union[Tensor, Sequence[Tensor]]:
    """Computes explained variance.

    Args:
        preds: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> from paddlemetrics.functional import explained_variance
        >>> target = B.tensor([3, -0.5, 2, 7])
        >>> preds = B.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = B.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = B.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance(preds, target, multioutput='raw_values')
        tensor([0.9677, 1.0000])
    """
    n_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(preds, target)
    return _explained_variance_compute(
        n_obs,
        sum_error,
        sum_squared_error,
        sum_target,
        sum_squared_target,
        multioutput,
    )
