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
import numpy as np

from paddlemetrics.utilities.imports import _PESQ_AVAILABLE

if _PESQ_AVAILABLE:
    import pesq as pesq_backend
else:
    pesq_backend = None
import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def pesq(preds: Tensor, target: Tensor, fs: int, mode: str, keep_same_device: bool = False) -> Tensor:
    r"""PESQ (Perceptual Evaluation of Speech Quality)

    This is a wrapper for the ``pesq`` package [1]. Note that input will be moved to `cpu`
    to perform the metric calculation.

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        paddlemetrics[audio]`` or ``pip install pesq``

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        fs:
            sampling frequency, should be 16000 or 8000 (Hz)
        mode:
            'wb' (wide-band) or 'nb' (narrow-band)
        keep_same_device:
            whether to move the pesq value to the device of preds

    Returns:
        pesq value of shape [...]

    Raises:
        ValueError:
            If ``peqs`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``

    Example:
        >>> from paddlemetrics.functional.audio import pesq
        >>> import torchapi as B
        >>> g = B.manual_seed(1)
        >>> preds = B.randn(8000)
        >>> target = B.randn(8000)
        >>> pesq(preds, target, 8000, 'nb')
        tensor(2.2076)
        >>> pesq(preds, target, 16000, 'wb')
        tensor(1.7359)

    References:
        [1] https://github.com/ludlows/python-pesq
    """
    if not _PESQ_AVAILABLE:
        raise ValueError(
            "PESQ metric requires that pesq is installed."
            "Either install as `pip install paddlemetrics[audio]` or `pip install pesq`"
        )
    if fs not in (8000, 16000):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
    if mode not in ("wb", "nb"):
        raise ValueError(f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        pesq_val_np = pesq_backend.pesq(fs, target.detach().cpu().numpy(), preds.detach().cpu().numpy(), mode)
        pesq_val = B.tensor(pesq_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        pesq_val_np = np.empty(shape=(preds_np.shape[0]))
        for b in range(preds_np.shape[0]):
            pesq_val_np[b] = pesq_backend.pesq(fs, target_np[b, :], preds_np[b, :], mode)
        pesq_val = B.from_numpy(pesq_val_np)
        pesq_val = pesq_val.reshape(preds.shape[:-1])

    if keep_same_device:
        pesq_val = pesq_val.to(preds.device)

    return pesq_val
