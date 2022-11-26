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
import warnings
from typing import Optional, Sequence, Tuple

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.functional.classification.auc import _auc_compute_without_check
from paddlemetrics.functional.classification.roc import roc
from paddlemetrics.utilities.checks import _input_format_classification
from paddlemetrics.utilities.enums import AverageMethod, DataType
from paddlemetrics.utilities.imports import _TORCH_LOWER_1_6


def _auroc_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, DataType]:
    """Updates and returns variables required to compute Area Under the Receiver Operating Characteristic Curve.
    Validates the inputs and returns the mode of the inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    # use _input_format_classification for validating the input and get the mode of data
    _, _, mode = _input_format_classification(preds, target)

    if mode == "multi class multi dim":
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.flatten()
    if mode == "multi-label" and preds.ndim > 2:
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)

    return preds, target, mode


def _auroc_compute(
    preds: Tensor,
    target: Tensor,
    mode: DataType,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
) -> Tensor:
    """Computes Area Under the Receiver Operating Characteristic Curve.

    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        mode: 'multi class multi dim' or 'multi-label' or 'binary'
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class.
            Should be set to ``None`` for binary problems
        average: Defines the reduction that is applied to the output:
        max_fpr: If not ``None``, calculates standardized partial AUC over the
            range [0, max_fpr]. Should be a float between 0 and 1.
        sample_weights: sample weights for each data point

    Example:
        >>> # binary case
        >>> preds = B.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = B.tensor([0, 0, 1, 1, 1])
        >>> preds, target, mode = _auroc_update(preds, target)
        >>> _auroc_compute(preds, target, mode, pos_label=1)
        tensor(0.5000)

        >>> # multiclass case
        >>> preds = B.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = B.tensor([0, 1, 1, 2, 2])
        >>> preds, target, mode = _auroc_update(preds, target)
        >>> _auroc_compute(preds, target, mode, num_classes=3)
        tensor(0.7778)
    """

    # binary mode override num_classes
    if mode == DataType.BINARY:
        num_classes = 1

    # check max_fpr parameter
    if max_fpr is not None:
        if not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
            raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

        if _TORCH_LOWER_1_6:
            raise RuntimeError(
                "`max_fpr` argument requires `B.bucketize` which" " is not available below PyTorch version 1.6"
            )

        # max_fpr parameter is only support for binary
        if mode != DataType.BINARY:
            raise ValueError(
                f"Partial AUC computation not available in"
                f" multilabel/multiclass setting, 'max_fpr' must be"
                f" set to `None`, received `{max_fpr}`."
            )

    # calculate fpr, tpr
    if mode == DataType.MULTILABEL:
        if average == AverageMethod.MICRO:
            fpr, tpr, _ = roc(preds.flatten(), target.flatten(), 1, pos_label, sample_weights)
        elif num_classes:
            # for multilabel we iteratively evaluate roc in a binary fashion
            output = [
                roc(preds[:, i], target[:, i], num_classes=1, pos_label=1, sample_weights=sample_weights)
                for i in range(num_classes)
            ]
            fpr = [o[0] for o in output]
            tpr = [o[1] for o in output]
        else:
            raise ValueError("Detected input to be `multilabel` but you did not provide `num_classes` argument")
    else:
        if mode != DataType.BINARY:
            if num_classes is None:
                raise ValueError("Detected input to `multiclass` but you did not provide `num_classes` argument")
            if average == AverageMethod.WEIGHTED and len(B.unique(target)) < num_classes:
                # If one or more classes has 0 observations, we should exclude them, as its weight will be 0
                target_bool_mat = B.zeros((len(target), num_classes), dtype=bool)
                target_bool_mat[B.arange(len(target)), target.long()] = 1
                class_observed = target_bool_mat.sum(axis=0) > 0
                for c in range(num_classes):
                    if not class_observed[c]:
                        warnings.warn(f"Class {c} had 0 observations, omitted from AUROC calculation", UserWarning)
                preds = preds[:, class_observed]
                target = target_bool_mat[:, class_observed]
                target = B.where(target)[1]
                num_classes = class_observed.sum()
                if num_classes == 1:
                    raise ValueError("Found 1 non-empty class in `multiclass` AUROC calculation")
        fpr, tpr, _ = roc(preds, target, num_classes, pos_label, sample_weights)

    # calculate standard roc auc score
    if max_fpr is None or max_fpr == 1:
        if mode == DataType.MULTILABEL and average == AverageMethod.MICRO:
            pass
        elif num_classes != 1:
            # calculate auc scores per class
            auc_scores = [_auc_compute_without_check(x, y, 1.0) for x, y in zip(fpr, tpr)]

            # calculate average
            if average == AverageMethod.NONE:
                return tensor(auc_scores)
            if average == AverageMethod.MACRO:
                return B.mean(B.stack(auc_scores))
            if average == AverageMethod.WEIGHTED:
                if mode == DataType.MULTILABEL:
                    support = B.sum(target, dim=0)
                else:
                    support = B.bincount(target.flatten(), minlength=num_classes)
                return B.sum(B.stack(auc_scores) * support / support.sum())

            allowed_average = (AverageMethod.NONE.value, AverageMethod.MACRO.value, AverageMethod.WEIGHTED.value)
            raise ValueError(
                f"Argument `average` expected to be one of the following:" f" {allowed_average} but got {average}"
            )

        return _auc_compute_without_check(fpr, tpr, 1.0)

    _device = fpr.device if isinstance(fpr, Tensor) else fpr[0].device
    max_area: Tensor = tensor(max_fpr, device=_device)
    # Add a single point at max_fpr and interpolate its tpr value
    stop = B.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = B.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = B.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = B.cat([fpr[:stop], max_area.view(1)])

    # Compute partial AUC
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)

    # McClish correction: standardize result to be 0.5 if non-discriminant and 1 if maximal
    min_area: Tensor = 0.5 * max_area ** 2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def auroc(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
) -> Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_)

    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        average:
            - ``'micro'`` computes metric globally. Only works for multilabel problems
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range [0, max_fpr]. Should be a float between 0 and 1.
        sample_weights: sample weights for each data point

    Raises:
        ValueError:
            If ``max_fpr`` is not a ``float`` in the range ``(0, 1]``.
        RuntimeError:
            If ``PyTorch version`` is ``below 1.6`` since max_fpr requires `B.bucketize`
            which is not available below 1.6.
        ValueError:
            If ``max_fpr`` is not set to ``None`` and the mode is ``not binary``
            since partial AUC computation is not available in multilabel/multiclass.
        ValueError:
            If ``average`` is none of ``None``, ``"macro"`` or ``"weighted"``.

    Example (binary case):
        >>> from paddlemetrics.functional import auroc
        >>> preds = B.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = B.tensor([0, 0, 1, 1, 1])
        >>> auroc(preds, target, pos_label=1)
        tensor(0.5000)

    Example (multiclass case):
        >>> preds = B.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = B.tensor([0, 1, 1, 2, 2])
        >>> auroc(preds, target, num_classes=3)
        tensor(0.7778)
    """
    preds, target, mode = _auroc_update(preds, target)
    return _auroc_compute(preds, target, mode, num_classes, pos_label, average, max_fpr, sample_weights)
