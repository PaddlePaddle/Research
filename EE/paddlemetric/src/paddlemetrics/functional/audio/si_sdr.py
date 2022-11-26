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
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def si_sdr(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not

    Returns:
        si-sdr value of shape [...]

    Example:
        >>> from paddlemetrics.functional.audio import si_sdr
        >>> target = B.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = B.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_sdr_val = si_sdr(preds, target)
        >>> si_sdr_val
        tensor(18.4030)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    _check_same_shape(preds, target)
    EPS = B.finfo(preds.dtype).eps

    if zero_mean:
        target = target - B.mean(target, dim=-1, keepdim=True)
        preds = preds - B.mean(preds, dim=-1, keepdim=True)

    alpha = (B.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        B.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    si_sdr_value = (B.sum(target_scaled ** 2, dim=-1) + EPS) / (B.sum(noise ** 2, dim=-1) + EPS)
    si_sdr_value = 10 * B.log10(si_sdr_value)

    return si_sdr_value
