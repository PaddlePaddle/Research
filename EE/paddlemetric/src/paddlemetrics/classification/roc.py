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
from typing import Any, Callable, List, Optional, Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.functional.classification.roc import _roc_compute, _roc_update
from paddlemetrics.metric import Metric
from paddlemetrics.utilities import rank_zero_warn


class ROC(Metric):
    """Computes the Receiver Operating Characteristic (ROC). Works for both binary, multiclass and multilabel
    problems. In the case of multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass/multilabel) tensor
      with probabilities, where C is the number of classes/labels.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Example (binary case):
        >>> from paddlemetrics import ROC
        >>> pred = B.tensor([0, 1, 2, 3])
        >>> target = B.tensor([0, 1, 1, 1])
        >>> roc = ROC(pos_label=1)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([4, 3, 2, 1, 0])

    Example (multiclass case):
        >>> pred = B.tensor([[0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75]])
        >>> target = B.tensor([0, 1, 3, 2])
        >>> roc = ROC(num_classes=4)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500]),
         tensor([1.7500, 0.7500, 0.0500])]

    Example (multilabel case):
        >>> pred = B.tensor([[0.8191, 0.3680, 0.1138],
        ...                      [0.3584, 0.7576, 0.1183],
        ...                      [0.2286, 0.3468, 0.1338],
        ...                      [0.8603, 0.0745, 0.1837]])
        >>> target = B.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> roc = ROC(num_classes=3, pos_label=1)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0., 0., 1., 1., 1.]),
         tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
         tensor([0., 1., 1., 1., 1.])]
        >>> thresholds # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.8603, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.7576, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.1837, 0.1837, 0.1338, 0.1183, 0.1138])]
    """

    is_differentiable = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
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

        self.num_classes = num_classes
        self.pos_label = pos_label

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            "Metric `ROC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _roc_update(preds, target, self.num_classes, self.pos_label)
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute the receiver operating characteristic.

        Returns:
            3-element tuple containing

            fpr:
                tensor with false positive rates.
                If multiclass, this is a list of such tensors, one for each class.
            tpr:
                tensor with true positive rates.
                If multiclass, this is a list of such tensors, one for each class.
            thresholds:
                thresholds used for computing false- and true postive rates
        """
        preds = B.cat(self.preds, dim=0)
        target = B.cat(self.target, dim=0)
        if not self.num_classes:
            raise ValueError(f"`num_classes` bas to be positive number, but got {self.num_classes}")
        return _roc_compute(preds, target, self.num_classes, self.pos_label)
