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

from paddlemetrics.utilities.checks import _check_retrieval_functional_inputs


def _dcg(target: Tensor) -> Tensor:
    """Computes Discounted Cumulative Gain for input tensor."""
    denom = B.log2(B.arange(target.shape[-1], device=target.device) + 2.0)
    return (target / denom).sum(dim=-1)


def retrieval_normalized_dcg(preds: Tensor, target: Tensor, k: Optional[int] = None) -> Tensor:
    """Computes `Normalized Discounted Cumulative Gain`_ (for information retrieval).

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be `float`,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        k: consider only the top k elements (default: None, which considers them all)

    Return:
        a single-value tensor with the nDCG of the predictions ``preds`` w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from paddlemetrics.functional import retrieval_normalized_dcg
        >>> preds = B.tensor([.1, .2, .3, 4, 70])
        >>> target = B.tensor([10, 0, 0, 1, 5])
        >>> retrieval_normalized_dcg(preds, target)
        tensor(0.6957)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    k = preds.shape[-1] if k is None else k

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer or None")

    sorted_target = target[B.argsort(preds, dim=-1, descending=True)][:k]
    ideal_target = B.sort(target, descending=True)[0][:k]

    ideal_dcg = _dcg(ideal_target)
    target_dcg = _dcg(sorted_target)

    # filter undefined scores
    all_irrelevant = ideal_dcg == 0
    target_dcg[all_irrelevant] = 0
    target_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]

    return target_dcg.mean()
