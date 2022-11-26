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
import paddleext.torchapi as B
from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_average_precision(preds: Tensor, target: Tensor) -> Tensor:
    """Computes average precision (for information retrieval), as explained in `IR Average precision`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be `float`,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.

    Return:
        a single-value tensor with the average precision (AP) of the predictions ``preds`` w.r.t. the labels ``target``.

    Example:
        >>> from paddlemetrics.functional import retrieval_average_precision
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_average_precision(preds, target)
        tensor(0.8333)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    if not target.sum():
        return tensor(0.0, device=preds.device)

    target = target[B.argsort(preds, dim=-1, descending=True)]
    positions = B.arange(1, len(target) + 1, device=target.device, dtype=B.float32)[target > 0]
    res = B.div((B.arange(len(positions), device=positions.device, dtype=B.float32) + 1), positions).mean()
    return res
