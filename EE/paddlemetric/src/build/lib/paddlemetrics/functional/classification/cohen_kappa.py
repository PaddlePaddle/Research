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
from typing import Optional

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.functional.classification.confusion_matrix import _confusion_matrix_compute, _confusion_matrix_update

_cohen_kappa_update = _confusion_matrix_update


def _cohen_kappa_compute(confmat: Tensor, weights: Optional[str] = None) -> Tensor:
    """Computes Cohen's kappa based on the weighting type.

    Args:
        confmat: Confusion matrix without normalization
        weights: Weighting type to calculate the score. Choose from
            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

    Example:
        >>> target = B.tensor([1, 1, 0, 0])
        >>> preds = B.tensor([0, 1, 0, 0])
        >>> confmat = _cohen_kappa_update(preds, target, num_classes=2)
        >>> _cohen_kappa_compute(confmat)
        tensor(0.5000)
    """

    confmat = _confusion_matrix_compute(confmat)
    confmat = confmat.float() if not confmat.is_floating_point() else confmat
    n_classes = confmat.shape[0]
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()  # outer product

    if weights is None:
        w_mat = B.ones_like(confmat).flatten()
        w_mat[:: n_classes + 1] = 0
        w_mat = w_mat.reshape(n_classes, n_classes)
    elif weights in ("linear", "quadratic"):
        w_mat = B.zeros_like(confmat)
        w_mat += B.arange(n_classes, dtype=w_mat.dtype, device=w_mat.device)
        if weights == "linear":
            w_mat = B.abs(w_mat - w_mat.T)
        else:
            w_mat = B.pow(w_mat - w_mat.T, 2.0)
    else:
        raise ValueError(
            f"Received {weights} for argument ``weights`` but should be either" " None, 'linear' or 'quadratic'"
        )

    k = B.sum(w_mat * confmat) / B.sum(w_mat * expected)
    return 1 - k


def cohen_kappa(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    weights: Optional[str] = None,
    threshold: float = 0.5,
) -> Tensor:
    r"""
    Calculates `Cohen's kappa score`_ that measures inter-annotator agreement.
     It is defined as

     .. math::
         \kappa = (p_o - p_e) / (1 - p_e)

     where :math:`p_o` is the empirical probability of agreement and :math:`p_e` isg
     the expected agreement when both annotators assign labels randomly. Note that
     :math:`p_e` is estimated using a per-annotator empirical prior over the
     class labels.

     Args:
         preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
             ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities

         target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels

         num_classes: Number of classes in the dataset.

         weights: Weighting type to calculate the score. Choose from
             - ``None`` or ``'none'``: no weighting
             - ``'linear'``: linear weighting
             - ``'quadratic'``: quadratic weighting

         threshold:
             Threshold value for binary or multi-label probabilities. default: 0.5

     Example:
         >>> from paddlemetrics.functional import cohen_kappa
         >>> target = B.tensor([1, 1, 0, 0])
         >>> preds = B.tensor([0, 1, 0, 0])
         >>> cohen_kappa(preds, target, num_classes=2)
         tensor(0.5000)
    """
    confmat = _cohen_kappa_update(preds, target, num_classes, threshold)
    return _cohen_kappa_compute(confmat, weights)
