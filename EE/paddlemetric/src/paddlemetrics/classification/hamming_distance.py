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

from paddlemetrics.functional.classification.hamming_distance import _hamming_distance_compute, _hamming_distance_update
from paddlemetrics.metric import Metric


class HammingDistance(Metric):
    r"""
    Computes the average `Hamming distance`_ (also
    known as Hamming loss) between targets and predictions:

    .. math::
        \text{Hamming distance} = \frac{1}{N \cdot L}\sum_i^N \sum_l^L 1(y_{il} \neq \hat{y_{il}})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label.

    Accepts all input types listed in :ref:`references/modules:input types`.

    Args:
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
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

    Raises:
        ValueError:
            If ``threshold`` is not between ``0`` and ``1``.

    Example:
        >>> from paddlemetrics import HammingDistance
        >>> target = B.tensor([[0, 1], [1, 1]])
        >>> preds = B.tensor([[0, 1], [0, 1]])
        >>> hamming_distance = HammingDistance()
        >>> hamming_distance(preds, target)
        tensor(0.2500)

    """
    is_differentiable = False
    correct: Tensor
    total: Tensor

    def __init__(
        self,
        threshold: float = 0.5,
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

        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.

        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth labels
        """
        correct, total = _hamming_distance_update(preds, target, self.threshold)

        self.correct += correct
        self.total += total

    def compute(self) -> Tensor:
        """Computes hamming distance based on inputs passed in to ``update`` previously."""
        return _hamming_distance_compute(self.correct, self.total)
