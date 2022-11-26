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

from paddlemetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from paddlemetrics.utilities.enums import AverageMethod as AvgMethod
from paddlemetrics.utilities.enums import MDMCAverageMethod


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """prevent zero division."""
    denom[denom == 0.0] = 1
    return num / denom


def _fbeta_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    beta: float,
    ignore_index: Optional[int],
    average: str,
    mdmc_average: Optional[str],
) -> Tensor:
    """Computes f_beta metric from stat scores: true positives, false positives, true negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        beta: The parameter `beta` (which determines the weight of recall in the combined score)
        ignore_index: Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)

    Example:
        >>> from paddlemetrics.functional.classification.stat_scores import _stat_scores_update
        >>> target = B.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = B.tensor([0, 2, 1, 0, 0, 1])
        >>> tp, fp, tn, fn = _stat_scores_update(
        ...                         preds,
        ...                         target,
        ...                         reduce='micro',
        ...                         num_classes=3,
        ...                     )
        >>> _fbeta_compute(tp, fp, tn, fn, beta=0.5, ignore_index=None, average='micro', mdmc_average=None)
        tensor(0.3333)
    """
    if average == AvgMethod.MICRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        mask = tp >= 0
        precision = _safe_divide(tp[mask].sum().float(), (tp[mask] + fp[mask]).sum())
        recall = _safe_divide(tp[mask].sum().float(), (tp[mask] + fn[mask]).sum())
    else:
        precision = _safe_divide(tp.float(), tp + fp)
        recall = _safe_divide(tp.float(), tp + fn)

    num = (1 + beta ** 2) * precision * recall
    denom = beta ** 2 * precision + recall
    denom[denom == 0.0] = 1.0  # avoid division by 0

    # if classes matter and a given class is not present in both the preds and the target,
    # computing the score for this class is meaningless, thus they should be ignored
    if average == AvgMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = B.nonzero((tp | fn | fp) == 0).cpu()
        if ignore_index is None:
            ignore_index = meaningless_indeces
        else:
            ignore_index = B.unique(B.cat((meaningless_indeces, B.tensor([[ignore_index]]))))

    if ignore_index is not None:
        if average not in (AvgMethod.MICRO, AvgMethod.SAMPLES) and mdmc_average == MDMCAverageMethod.SAMPLEWISE:
            num[..., ignore_index] = -1
            denom[..., ignore_index] = -1
        elif average not in (AvgMethod.MICRO, AvgMethod.SAMPLES):
            num[ignore_index, ...] = -1
            denom[ignore_index, ...] = -1

    if average == AvgMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = (tp + fp + fn == 0) | (tp + fp + fn == -3)
        num = num[~cond]
        denom = denom[~cond]

    return _reduce_stat_scores(
        numerator=num,
        denominator=denom,
        weights=None if average != AvgMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def fbeta(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
) -> Tensor:
    r"""
    Computes f_beta metric.

    .. math::
        F_{\beta} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities or logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label logits or probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        average:
            Defines the reduction that is applied. Should be one of the following:

                - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
                - ``'macro'``: Calculate the metric for each class separately, and average the
                  metrics across classes (with equal weights for each class).
                - ``'weighted'``: Calculate the metric for each class separately, and average the
                  metrics across classes, weighting each class by its support (``tp + fn``).
                - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
                  the metric for every class.
                - ``'samples'``: Calculate the metric for each sample, and average the metrics
                  across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the `preds` or `target`,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

                - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
                  multi-class.
                - ``'samplewise'``: In this case, the statistics are computed separately for each
                  sample on the ``N`` axis, and then averaged over samples.
                  The computation for each sample is done by treating the flattened extra axes ``...``
                  (see :ref:`references/modules:input types`) as the ``N`` dimension within the sample,
                  and computing the metric for the sample based on that.
                - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
                  (see :ref:`references/modules:input types`)
                  are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
                  were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from paddlemetrics.functional import fbeta
        >>> target = B.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = B.tensor([0, 2, 1, 0, 0, 1])
        >>> fbeta(preds, target, num_classes=3, beta=0.5)
        tensor(0.3333)

    """
    allowed_average = list(AvgMethod)
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if mdmc_average is not None and MDMCAverageMethod.from_str(mdmc_average) is None:
        raise ValueError(f"The `mdmc_average` has to be one of {list(MDMCAverageMethod)}, got {mdmc_average}.")

    if average in [AvgMethod.MACRO, AvgMethod.WEIGHTED, AvgMethod.NONE] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    reduce = AvgMethod.MACRO if average in [AvgMethod.WEIGHTED, AvgMethod.NONE] else average
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_average,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _fbeta_compute(tp, fp, tn, fn, beta, ignore_index, average, mdmc_average)


def f1(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
) -> Tensor:
    """Computes F1 metric. F1 metrics correspond to a equally weighted average of the precision and recall scores.

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities or logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the `preds` or `target`,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`references/modules:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`references/modules:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from paddlemetrics.functional import f1
        >>> target = B.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = B.tensor([0, 2, 1, 0, 0, 1])
        >>> f1(preds, target, num_classes=3)
        tensor(0.3333)
    """
    return fbeta(preds, target, 1.0, average, mdmc_average, ignore_index, num_classes, threshold, top_k, multiclass)
