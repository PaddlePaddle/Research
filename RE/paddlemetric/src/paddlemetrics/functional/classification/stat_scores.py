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
from typing import List, Optional, Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.utilities.checks import _input_format_classification
from paddlemetrics.utilities.enums import AverageMethod, MDMCAverageMethod


def _del_column(data: Tensor, idx: int) -> Tensor:
    """Delete the column at index."""
    return B.cat([data[:, :idx], data[:, (idx + 1) :]], 1)


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds:
            An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target:
            An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce:
            One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depnds on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then
        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce'samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then
        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors
    """
    dim: Union[int, List[int]] = 1  # for "samples"
    if reduce == "micro":
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)
    return tp.long(), fp.long(), tn.long(), fn.long()


def _stat_scores_update(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.5,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns the the number of true positives, false positives, true negatives, false negatives.
    Raises ValueError if:

        - The `ignore_index` is not valid
        - When `ignore_index` is used with binary data
        - When inputs are multi-dimensional multi-class, and the `mdmc_reduce` parameter is not set

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied
        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handeled
        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        top_k: Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
    """

    preds, target, _ = _input_format_classification(
        preds, target, threshold=threshold, num_classes=num_classes, multiclass=multiclass, top_k=top_k
    )

    if ignore_index is not None and not 0 <= ignore_index < preds.shape[1]:
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {preds.shape[0]} classes")

    if ignore_index is not None and preds.shape[1] == 1:
        raise ValueError("You can not use `ignore_index` with binary data.")

    if preds.ndim == 3:
        if not mdmc_reduce:
            raise ValueError(
                "When your inputs are multi-dimensional multi-class, you have to set the `mdmc_reduce` parameter"
            )
        if mdmc_reduce == "global":
            preds = B.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
            target = B.transpose(target, 1, 2).reshape(-1, target.shape[1])

    # Delete what is in ignore_index, if applicable (and classes don't matter):
    if ignore_index is not None and reduce != "macro":
        preds = _del_column(preds, ignore_index)
        target = _del_column(target, ignore_index)

    tp, fp, tn, fn = _stat_scores(preds, target, reduce=reduce)

    # Take care of ignore_index
    if ignore_index is not None and reduce == "macro":
        tp[..., ignore_index] = -1
        fp[..., ignore_index] = -1
        tn[..., ignore_index] = -1
        fn[..., ignore_index] = -1

    return tp, fp, tn, fn


def _stat_scores_compute(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    """Computes the number of true positives, false positives, true negatives, false negatives. Concatenates the
    input tensors along with the support into one output.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives

    Example:
        >>> preds  = B.tensor([1, 0, 2, 1])
        >>> target = B.tensor([1, 1, 2, 0])
        >>> tp, fp, tn, fn = _stat_scores_update(preds, target, reduce='macro', num_classes=3)
        >>> _stat_scores_compute(tp, fp, tn, fn)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> tp, fp, tn, fn = _stat_scores_update(preds, target, reduce='micro')
        >>> _stat_scores_compute(tp, fp, tn, fn)
        tensor([2, 2, 6, 2, 4])
    """
    stats = [
        tp.unsqueeze(-1),
        fp.unsqueeze(-1),
        tn.unsqueeze(-1),
        fn.unsqueeze(-1),
        tp.unsqueeze(-1) + fn.unsqueeze(-1),  # support
    ]
    outputs: Tensor = B.cat(stats, -1)
    outputs = B.where(outputs < 0, tensor(-1, device=outputs.device, dtype=outputs.dtype), outputs)

    return outputs


def _reduce_stat_scores(
    numerator: Tensor,
    denominator: Tensor,
    weights: Optional[Tensor],
    average: Optional[str],
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Reduces scores of type ``numerator/denominator`` or.

    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights: A tensor of weights to be used if ``average='weighted'``.
        average: The method to average the scores
        mdmc_average: The method to average the scores if inputs were multi-dimensional multi-class (MDMC)
        zero_division: The value to use for the score if denominator equals zero.
    """
    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    if weights is None:
        weights = B.ones_like(denominator)
    else:
        weights = weights.float()

    numerator = B.where(zero_div_mask, tensor(float(zero_division), device=numerator.device), numerator)
    denominator = B.where(zero_div_mask | ignore_mask, tensor(1.0, device=denominator.device), denominator)
    weights = B.where(ignore_mask, tensor(0.0, device=weights.device), weights)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)

    scores = weights * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = B.where(B.isnan(scores), tensor(float(zero_division), device=scores.device), scores)

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()

    if average in (AverageMethod.NONE, None):
        scores = B.where(ignore_mask, tensor(float("nan"), device=scores.device), scores)
    else:
        scores = scores.sum()

    return scores


def stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: str = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.5,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Computes the number of true positives, false positives, true negatives, false negatives.
    Related to `Type I and Type II errors`_
    and the `confusion matrix`_.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        reduce:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all [sample, class]
              combinations (globally). Each statistic is represented by a single integer.
            - ``'macro'``: Counts the statistics for each class separately (over all samples).
              Each statistic is represented by a ``(C,)`` tensor. Requires ``num_classes``
              to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Each statistic is represented by a ``(N, )`` 1d tensor.

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_reduce``.

        num_classes:
            Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.

        ignore_index:
            Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.

        mdmc_reduce:
            Defines how the multi-dimensional multi-class inputs are handeled. Should be
            one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class (see :ref:`references/modules:input types` for the definition of input types).

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then the outputs are concatenated together. In each
              sample the extra axes ``...`` are flattened to become the sub-sample axis, and
              statistics for each sample are computed by treating the sub-sample axis as the
              ``N`` axis for that sample.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs are
              flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``reduce`` parameter applies as usual.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The
        shape depends on the ``reduce`` and ``mdmc_reduce`` (in case of multi-dimensional
        multi-class data) parameters:

        - If the data is not multi-dimensional multi-class, then

          - If ``reduce='micro'``, the shape will be ``(5, )``
          - If ``reduce='macro'``, the shape will be ``(C, 5)``,
            where ``C`` stands for the number of classes
          - If ``reduce='samples'``, the shape will be ``(N, 5)``, where ``N`` stands for
            the number of samples

        - If the data is multi-dimensional multi-class and ``mdmc_reduce='global'``, then

          - If ``reduce='micro'``, the shape will be ``(5, )``
          - If ``reduce='macro'``, the shape will be ``(C, 5)``
          - If ``reduce='samples'``, the shape will be ``(N*X, 5)``, where ``X`` stands for
            the product of sizes of all "extra" dimensions of the data (i.e. all dimensions
            except for ``C`` and ``N``)

        - If the data is multi-dimensional multi-class and ``mdmc_reduce='samplewise'``, then

          - If ``reduce='micro'``, the shape will be ``(N, 5)``
          - If ``reduce='macro'``, the shape will be ``(N, C, 5)``
          - If ``reduce='samples'``, the shape will be ``(N, X, 5)``

    Raises:
        ValueError:
            If ``reduce`` is none of ``"micro"``, ``"macro"`` or ``"samples"``.
        ValueError:
            If ``mdmc_reduce`` is none of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``reduce`` is set to ``"macro"`` and ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``ignore_index`` is used with ``binary data``.
        ValueError:
            If inputs are ``multi-dimensional multi-class`` and ``mdmc_reduce`` is not provided.

    Example:
        >>> from paddlemetrics.functional import stat_scores
        >>> preds  = B.tensor([1, 0, 2, 1])
        >>> target = B.tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, reduce='macro', num_classes=3)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> stat_scores(preds, target, reduce='micro')
        tensor([2, 2, 6, 2, 4])

    """
    if reduce not in ["micro", "macro", "samples"]:
        raise ValueError(f"The `reduce` {reduce} is not valid.")

    if mdmc_reduce not in [None, "samplewise", "global"]:
        raise ValueError(f"The `mdmc_reduce` {mdmc_reduce} is not valid.")

    if reduce == "macro" and (not num_classes or num_classes < 1):
        raise ValueError("When you set `reduce` as 'macro', you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_reduce,
        top_k=top_k,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )
    return _stat_scores_compute(tp, fp, tn, fn)
