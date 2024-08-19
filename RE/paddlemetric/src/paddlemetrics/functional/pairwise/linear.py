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

from paddleext.torchapi import  Tensor

from paddlemetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix


def _pairwise_linear_similarity_update(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculates the pairwise linear similarity matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero
    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)

    distance = x @ y.T
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance


def pairwise_linear_similarity(
    x: Tensor, y: Optional[Tensor] = None, reduction: Optional[str] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    r"""
    Calculates pairwise linear similarity:

    .. math::
        s_{lin}(x,y) = <x,y> = \sum_{d=1}^D x_d \cdot y_d

    If both `x` and `y` are passed in, the calculation will be performed pairwise between the rows of `x` and `y`.
    If only `x` is passed in, the calculation will be performed between the rows of `x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torchapi as B
        >>> from paddlemetrics.functional import pairwise_linear_similarity
        >>> x = B.tensor([[2, 3], [3, 5], [5, 8]], dtype=B.float32)
        >>> y = B.tensor([[1, 0], [2, 1]], dtype=B.float32)
        >>> pairwise_linear_similarity(x, y)
        tensor([[ 2.,  7.],
                [ 3., 11.],
                [ 5., 18.]])
        >>> pairwise_linear_similarity(x)
        tensor([[ 0., 21., 34.],
                [21.,  0., 55.],
                [34., 55.,  0.]])

    """
    distance = _pairwise_linear_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
