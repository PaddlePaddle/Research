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
from paddleext.torchapi import  Tensor

from paddlemetrics.functional.regression.tweedie_deviance import (
    _tweedie_deviance_score_compute,
    _tweedie_deviance_score_update,
)
from paddlemetrics.metric import Metric


class TweedieDevianceScore(Metric):
    r"""
    Computes the `Tweedie Deviance Score`_ between targets and predictions:

    .. math::
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }power=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }power=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }power=2\\
        2 * (\frac{(max(y,0))^{2}}{(1 - power)(2 - power)} - \frac{y(\hat{y})^{1 - power}}{1 - power} + \frac{(\hat{y})
            ^{2 - power}}{2 - power}), & \text{otherwise}
        \end{cases}

    where :math:`y` is a tensor of targets values, and :math:`\hat{y}` is a tensor of predictions.

    Forward accepts

    - ``preds`` (float tensor): ``(N,...)``
    - ``targets`` (float tensor): ``(N,...)``

    Args:
        power:
            - power < 0 : Extreme stable distribution. (Requires: preds > 0.)
            - power = 0 : Normal distribution. (Requires: targets and preds can be any real numbers.)
            - power = 1 : Poisson distribution. (Requires: targets >= 0 and y_pred > 0.)
            - 1 < p < 2 : Compound Poisson distribution. (Requires: targets >= 0 and preds > 0.)
            - power = 2 : Gamma distribution. (Requires: targets > 0 and preds > 0.)
            - power = 3 : Inverse Gaussian distribution. (Requires: targets > 0 and preds > 0.)
            - otherwise : Positive stable distribution. (Requires: targets > 0 and preds > 0.)
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the all gather.

    Example:
        >>> from paddlemetrics import TweedieDevianceScore
        >>> targets = B.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = B.tensor([4.0, 3.0, 2.0, 1.0])
        >>> deviance_score = TweedieDevianceScore(power=2)
        >>> deviance_score(preds, targets)
        tensor(1.2083)

    """
    is_differentiable = True
    sum_deviance_score: Tensor
    num_observations: Tensor

    def __init__(
        self,
        power: float = 0.0,
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
        if 0 < power < 1:
            raise ValueError(f"Deviance Score is not defined for power={power}.")

        self.power: float = power

        self.add_state("sum_deviance_score", B.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_observations", B.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:  # type: ignore
        """Update metric states with predictions and targets.

        Args:
            preds: Predicted tensor with shape ``(N,d)``
            targets: Ground truth tensor with shape ``(N,d)``
        """
        sum_deviance_score, num_observations = _tweedie_deviance_score_update(preds, targets, self.power)

        self.sum_deviance_score += sum_deviance_score
        self.num_observations += num_observations

    def compute(self) -> Tensor:
        return _tweedie_deviance_score_compute(self.sum_deviance_score, self.num_observations)
