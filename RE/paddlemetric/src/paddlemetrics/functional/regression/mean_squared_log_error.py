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
from typing import Tuple

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def _mean_squared_log_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Returns variables required to compute Mean Squared Log Error. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    _check_same_shape(preds, target)
    sum_squared_log_error = B.sum(B.pow(B.log1p(preds) - B.log1p(target), 2))
    n_obs = target.numel()
    return sum_squared_log_error, n_obs


def _mean_squared_log_error_compute(sum_squared_log_error: Tensor, n_obs: int) -> Tensor:
    """Computes Mean Squared Log Error.

    Args:
        sum_squared_log_error: Sum of square of log errors over all observations
                                (log error = log(target) - log(prediction))
        n_obs: Number of predictions or observations

    Example:
        >>> preds = B.tensor([0., 1, 2, 3])
        >>> target = B.tensor([0., 1, 2, 2])
        >>> sum_squared_log_error, n_obs = _mean_squared_log_error_update(preds, target)
        >>> _mean_squared_log_error_compute(sum_squared_log_error, n_obs)
        tensor(0.0207)
    """

    return sum_squared_log_error / n_obs


def mean_squared_log_error(preds: Tensor, target: Tensor) -> Tensor:
    """Computes mean squared log error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with RMSLE

    Example:
        >>> from paddlemetrics.functional import mean_squared_log_error
        >>> x = B.tensor([0., 1, 2, 3])
        >>> y = B.tensor([0., 1, 2, 2])
        >>> mean_squared_log_error(x, y)
        tensor(0.0207)

    .. note::
        Half precision is only support on GPU for this metric
    """
    sum_squared_log_error, n_obs = _mean_squared_log_error_update(preds, target)
    return _mean_squared_log_error_compute(sum_squared_log_error, n_obs)
