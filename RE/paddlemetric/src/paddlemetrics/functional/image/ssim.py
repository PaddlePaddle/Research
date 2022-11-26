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
from typing import Optional, Sequence, Tuple

import paddleext.torchapi as B
from paddleext.torchapi import Tensor

from paddlemetrics.utilities.checks import _check_same_shape
from paddlemetrics.utilities.distributed import reduce


def _gaussian(kernel_size: int, sigma: float, dtype: B.dtype, device: B.device) -> Tensor:
    """Computes 1D gaussian kernel.

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    Example:
        >>> _gaussian(3, 1, B.float, 'cpu')
        tensor([[0.2741, 0.4519, 0.2741]])
    """
    dist = B.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=dtype, device=device)
    gauss = B.exp(-B.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)


def _gaussian_kernel(
    channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: B.dtype, device: B.device
) -> Tensor:
    """Computes 2D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    Example:
        >>> _gaussian_kernel(1, (5,5), (1,1), B.float, "cpu")
        tensor([[[[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]]]])
    """

    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = B.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


def _ssim_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Structural Similarity Index Measure. Checks for same shape
    and type of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ssim_compute(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Computes Structual Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03

    Example:
        >>> preds = B.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _ssim_update(preds, target)
        >>> _ssim_compute(preds, target)
        tensor(0.9219)
    """
    if len(kernel_size) != 2 or len(sigma) != 2:
        raise ValueError(
            "Expected `kernel_size` and `sigma` to have the length of two."
            f" Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())

    c1 = pow(k1 * data_range, 2)
    c2 = pow(k2 * data_range, 2)
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    kernel = _gaussian_kernel(channel, kernel_size, sigma, dtype, device)
    pad_h = (kernel_size[0] - 1) // 2
    pad_w = (kernel_size[1] - 1) // 2

    preds = B.pad(preds, (pad_h, pad_h, pad_w, pad_w), mode="reflect")
    target = B.pad(target, (pad_h, pad_h, pad_w, pad_w), mode="reflect")

    input_list = B.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)
    outputs = B.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2

    ssim_idx = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    ssim_idx = ssim_idx[..., pad_h:-pad_h, pad_w:-pad_w]

    return reduce(ssim_idx, reduction)


def ssim(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Computes Structual Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03

    Return:
        Tensor with SSIM score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` or ``sigma`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.
        ValueError:
            If one of the elements of ``sigma`` is not a ``positive number``.

    Example:
        >>> from paddlemetrics.functional import ssim
        >>> preds = B.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim(preds, target)
        tensor(0.9219)
    """
    preds, target = _ssim_update(preds, target)
    return _ssim_compute(preds, target, kernel_size, sigma, reduction, data_range, k1, k2)
