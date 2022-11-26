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
from typing import Any, Callable, Optional

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.functional.regression.mean_absolute_percentage_error import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from paddlemetrics.metric import Metric


class MeanAbsolutePercentageError(Metric):
    r"""
    Computes `Mean Absolute Percentage Error`_ (MAPE):

    .. math:: \text{MAPE} = \frac{1}{n}\sum_1^n\frac{|   y_i - \hat{y_i} |}{\max(\epsilon, y_i)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Note:
        The epsilon value is taken from `scikit-learn's implementation of MAPE`_.

    Note:
        MAPE output is a non-negative floating point. Best result is 0.0 . But it is important to note that,
        bad predictions, can lead to arbitarily large values. Especially when some ``target`` values are close to 0.
        This `MAPE implementation returns`_ a very large number instead of ``inf``.

    Example:
        >>> from paddlemetrics import MeanAbsolutePercentageError
        >>> target = B.tensor([1, 10, 1e6])
        >>> preds = B.tensor([0.9, 15, 1.2e6])
        >>> mean_abs_percentage_error = MeanAbsolutePercentageError()
        >>> mean_abs_percentage_error(preds, target)
        tensor(0.2667)

    """
    is_differentiable = True
    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Computes mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)
