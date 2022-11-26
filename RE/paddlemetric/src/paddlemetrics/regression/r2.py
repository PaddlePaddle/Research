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

from paddlemetrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
from paddlemetrics.metric import Metric


class R2Score(Metric):
    r"""
    Computes r2 score also known as `R2 Score_Coefficient Determination`_:

    .. math:: R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i - \bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the `adjusted` argument.

    Forward accepts

    - ``preds`` (float tensor): ``(N,)`` or ``(N, M)`` (multioutput)
    - ``target`` (float tensor): ``(N,)`` or ``(N, M)`` (multioutput)

    In the case of multioutput, as default the variances will be uniformly
    averaged over the additional dimensions. Please see argument `multioutput`
    for changing this behavior.

    Args:
        num_outputs:
            Number of outputs in multioutput setting (default is 1)
        adjusted:
            number of independent regressors for calculating adjusted r2 score.
            Default 0 (standard r2 score).
        multioutput:
            Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is ``'uniform_average'``.):

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Raises:
        ValueError:
            If ``adjusted`` parameter is not an integer larger or equal to 0.
        ValueError:
            If ``multioutput`` is not one of ``"raw_values"``, ``"uniform_average"`` or ``"variance_weighted"``.

    Example:
        >>> from paddlemetrics import R2Score
        >>> target = B.tensor([3, -0.5, 2, 7])
        >>> preds = B.tensor([2.5, 0.0, 2, 8])
        >>> r2score = R2Score()
        >>> r2score(preds, target)
        tensor(0.9486)

        >>> target = B.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = B.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2score = R2Score(num_outputs=2, multioutput='raw_values')
        >>> r2score(preds, target)
        tensor([0.9654, 0.9082])

    """
    is_differentiable = True
    sum_squared_error: Tensor
    sum_error: Tensor
    residual: Tensor
    total: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        adjusted: int = 0,
        multioutput: str = "uniform_average",
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

        self.num_outputs = num_outputs

        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError("`adjusted` parameter should be an integer larger or equal to 0.")
        self.adjusted = adjusted

        allowed_multioutput = ("raw_values", "uniform_average", "variance_weighted")
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput = multioutput

        self.add_state("sum_squared_error", default=B.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_error", default=B.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("residual", default=B.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, sum_error, residual, total = _r2_score_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total

    def compute(self) -> Tensor:
        """Computes r2 score over the metric states."""
        return _r2_score_compute(
            self.sum_squared_error, self.sum_error, self.residual, self.total, self.adjusted, self.multioutput
        )
