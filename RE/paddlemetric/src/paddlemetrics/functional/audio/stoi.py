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
import paddleext.torchapi as B

from paddlemetrics.utilities.imports import _PYSTOI_AVAILABLE

if _PYSTOI_AVAILABLE:
    from pystoi import stoi as stoi_backend
else:
    stoi_backend = None
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def stoi(preds: Tensor, target: Tensor, fs: int, extended: bool = False, keep_same_device: bool = False) -> Tensor:
    r"""STOI (Short Term Objective Intelligibility, see [2,3]), a wrapper for the pystoi package [1].
    Note that input will be moved to `cpu` to perform the metric calculation.

    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due
    to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations.
    The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good
    alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are
    interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms,
    on speech intelligibility. Description taken from [Cees Taal's website](http://www.ceestaal.nl/code/).

    .. note:: using this metrics requires you to have ``pystoi`` install. Either install as ``pip install
        paddlemetrics[audio]`` or ``pip install pystoi``

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        fs:
            sampling frequency (Hz)
        extended:
            whether to use the extended STOI described in [4]
        keep_same_device:
            whether to move the stoi value to the device of preds

    Returns:
        stoi value of shape [...]

    Raises:
        ValueError:
            If ``pystoi`` package is not installed

    Example:
        >>> from paddlemetrics.functional.audio import stoi
        >>> import torchapi as B
        >>> g = B.manual_seed(1)
        >>> preds = B.randn(8000)
        >>> target = B.randn(8000)
        >>> stoi(preds, target, 8000).float()
        tensor(-0.0100)

    References:
        [1] https://github.com/mpariente/pystoi

        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for
        Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.

        [3] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for Intelligibility Prediction of
        Time-Frequency Weighted Noisy Speech', IEEE Transactions on Audio, Speech, and Language Processing, 2011.

        [4] J. Jensen and C. H. Taal, 'An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated
        Noise Maskers', IEEE Transactions on Audio, Speech and Language Processing, 2016.

    """
    if not _PYSTOI_AVAILABLE:
        raise ValueError(
            "STOI metric requires that pystoi is installed."
            "Either install as `pip install paddlemetrics[audio]` or `pip install pystoi`"
        )
    _check_same_shape(preds, target)

    if len(preds.shape) == 1:
        stoi_val_np = stoi_backend(target.detach().cpu().numpy(), preds.detach().cpu().numpy(), fs, extended)
        stoi_val = B.tensor(stoi_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        stoi_val_np = np.empty(shape=(preds_np.shape[0]))
        for b in range(preds_np.shape[0]):
            stoi_val_np[b] = stoi_backend(target_np[b, :], preds_np[b, :], fs, extended)
        stoi_val = B.from_numpy(stoi_val_np)
        stoi_val = stoi_val.reshape(preds.shape[:-1])

    if keep_same_device:
        stoi_val = stoi_val.to(preds.device)

    return stoi_val
