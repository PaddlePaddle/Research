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

from paddlemetrics.functional.regression.mean_absolute_error import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)
from paddlemetrics.metric import Metric


class MeanAbsoluteError(Metric):
    r"""
    `Computes Mean Absolute Error`_ (MAE):

    .. math:: \text{MAE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} |

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:
        >>> from paddlemetrics import MeanAbsoluteError
        >>> target = B.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = B.tensor([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error = MeanAbsoluteError()
        >>> mean_absolute_error(preds, target)
        tensor(0.5000)
    """
    is_differentiable = True
    sum_abs_error: Tensor
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

        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)
