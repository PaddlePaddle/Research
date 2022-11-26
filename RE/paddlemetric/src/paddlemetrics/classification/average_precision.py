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
from typing import Any, List, Optional, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.functional.classification.average_precision import (
    _average_precision_compute,
    _average_precision_update,
)
from paddlemetrics.metric import Metric
from paddlemetrics.utilities import rank_zero_warn
from paddlemetrics.utilities.data import dim_zero_cat


class AveragePrecision(Metric):
    """Computes the average precision score, which summarises the precision recall curve into one number. Works for
    both binary and multiclass problems. In the case of multiclass, the values will be calculated based on a one-
    vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        average:
            defines the reduction that is applied in the case of multiclass and multilabel input.
            Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes. Cannot be
              used with multiclass input.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support.
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example (binary case):
        >>> from paddlemetrics import AveragePrecision
        >>> pred = B.tensor([0, 1, 2, 3])
        >>> target = B.tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(pos_label=1)
        >>> average_precision(pred, target)
        tensor(1.)

    Example (multiclass case):
        >>> pred = B.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = B.tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(num_classes=5, average=None)
        >>> average_precision(pred, target)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    is_differentiable = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.pos_label = pos_label
        allowed_average = ("micro", "macro", "weighted", None)
        if average not in allowed_average:
            raise ValueError(f"Expected argument `average` to be one of {allowed_average}" f" but got {average}")
        self.average = average

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AveragePrecision` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _average_precision_update(
            preds, target, self.num_classes, self.pos_label, self.average
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[Tensor, List[Tensor]]:
        """Compute the average precision score.

        Returns:
            tensor with average precision. If multiclass will return list
            of such tensors, one for each class
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self.num_classes:
            raise ValueError(f"`num_classes` bas to be positive number, but got {self.num_classes}")
        return _average_precision_compute(preds, target, self.num_classes, self.pos_label, self.average)
