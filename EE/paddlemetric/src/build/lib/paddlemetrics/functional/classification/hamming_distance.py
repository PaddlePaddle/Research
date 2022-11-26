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
from typing import Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _input_format_classification


def _hamming_distance_update(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tuple[Tensor, int]:
    """Returns the number of positions where prediction equals target, and number of predictions.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
    """

    preds, target, _ = _input_format_classification(preds, target, threshold=threshold)

    correct = (preds == target).sum()
    total = preds.numel()

    return correct, total


def _hamming_distance_compute(correct: Tensor, total: Union[int, Tensor]) -> Tensor:
    """Computes the Hamming distance.

    Args:
        correct: Number of positions where prediction equals target
        total: Total number of predictions

    Example:
        >>> target = B.tensor([[0, 1], [1, 1]])
        >>> preds = B.tensor([[0, 1], [0, 1]])
        >>> correct, total = _hamming_distance_update(preds, target)
        >>> _hamming_distance_compute(correct, total)
        tensor(0.2500)
    """

    return 1 - correct.float() / total


def hamming_distance(preds: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    r"""
    Computes the average `Hamming distance`_ (also
    known as Hamming loss) between targets and predictions:

    .. math::
        \text{Hamming distance} = \frac{1}{N \cdot L} \sum_i^N \sum_l^L 1(y_{il} \neq \hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label.

    Accepts all input types listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

    Example:
        >>> from paddlemetrics.functional import hamming_distance
        >>> target = B.tensor([[0, 1], [1, 1]])
        >>> preds = B.tensor([[0, 1], [0, 1]])
        >>> hamming_distance(preds, target)
        tensor(0.2500)

    """

    correct, total = _hamming_distance_update(preds, target, threshold)
    return _hamming_distance_compute(correct, total)
