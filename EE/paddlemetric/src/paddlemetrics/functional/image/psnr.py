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
from typing import Optional, Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.utilities import rank_zero_warn, reduce


def _psnr_compute(
    sum_squared_error: Tensor,
    n_obs: Tensor,
    data_range: Tensor,
    base: float = 10.0,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Computes peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        data_range:
            the range of the data. If None, it is determined from the data (max - min). ``data_range`` must be given
            when ``dim`` is not None.
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> preds = B.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = B.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> data_range = target.max() - target.min()
        >>> sum_squared_error, n_obs = _psnr_update(preds, target)
        >>> _psnr_compute(sum_squared_error, n_obs, data_range)
        tensor(2.5527)
    """

    psnr_base_e = 2 * B.log(data_range) - B.log(sum_squared_error / n_obs)
    psnr_vals = psnr_base_e * (10 / B.log(tensor(base)))
    return reduce(psnr_vals, reduction=reduction)


def _psnr_update(
    preds: Tensor,
    target: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.
    """

    if dim is None:
        sum_squared_error = B.sum(B.pow(preds - target, 2))
        n_obs = tensor(target.numel(), device=target.device)
        return sum_squared_error, n_obs

    diff = preds - target
    sum_squared_error = B.sum(diff * diff, dim=dim)

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    if not dim_list:
        n_obs = tensor(target.numel(), device=target.device)
    else:
        n_obs = tensor(target.size(), device=target.device)[dim_list].prod()
        n_obs = n_obs.expand_as(sum_squared_error)

    return sum_squared_error, n_obs


def psnr(
    preds: Tensor,
    target: Tensor,
    data_range: Optional[float] = None,
    base: float = 10.0,
    reduction: str = "elementwise_mean",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tensor:
    """Computes the peak signal-to-noise ratio.

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range:
            the range of the data. If None, it is determined from the data (max - min). ``data_range`` must be given
            when ``dim`` is not None.
        base: a base of a logarithm to use (default: 10)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.
    Return:
        Tensor with PSNR score

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not provided.

    Example:
        >>> from paddlemetrics.functional import psnr
        >>> pred = B.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = B.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(pred, target)
        tensor(2.5527)

    .. note::
        Half precision is only support on GPU for this metric
    """
    if dim is None and reduction != "elementwise_mean":
        rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

    if data_range is None:
        if dim is not None:
            # Maybe we could use `B.amax(target, dim=dim) - B.amin(target, dim=dim)` in PyTorch 1.7 to calculate
            # `data_range` in the future.
            raise ValueError("The `data_range` must be given when `dim` is not None.")

        data_range = target.max() - target.min()
    else:
        data_range = tensor(float(data_range))
    sum_squared_error, n_obs = _psnr_update(preds, target, dim=dim)
    return _psnr_compute(sum_squared_error, n_obs, data_range, base=base, reduction=reduction)
