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

from paddlemetrics.utilities.data import to_categorical
from paddlemetrics.utilities.distributed import reduce


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates the number of true positive, false positive, true negative and false negative for a specific
    class.

    Args:
        preds: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:
        >>> x = B.tensor([1, 2, 3])
        >>> y = B.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = _stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))
    """
    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(B.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(B.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(B.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(B.long).sum()
    sup = (target == class_index).to(B.long).sum()

    return tp, fp, tn, fn, sup


def dice_score(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Compute dice score from prediction scores.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor containing dice score

    Example:
        >>> from paddlemetrics.functional import dice_score
        >>> pred = B.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = B.tensor([0, 1, 3, 2])
        >>> dice_score(pred, target)
        tensor(0.3333)
    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = B.zeros(num_classes - bg_inv, device=preds.device, dtype=B.float32)
    for i in range(bg_inv, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg_inv] += no_fg_score
            continue

        # TODO: rewrite to use general `stat_scores`
        tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(B.float)
        # nan result
        score_cls = (2 * tp).to(B.float) / denom if B.is_nonzero(denom) else nan_score
        scores[i - bg_inv] += score_cls.item()

    return reduce(scores, reduction=reduction)
