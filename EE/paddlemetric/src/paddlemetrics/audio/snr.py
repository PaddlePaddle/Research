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
from typing import Any, Callable, Optional

from paddleext.torchapi import  Tensor, tensor

from paddlemetrics.functional.audio.snr import snr
from paddlemetrics.metric import Metric


class SNR(Metric):
    r"""Signal-to-noise ratio (SNR_):

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level
    of the desired signal to the level of background noise. Therefore, a high value of
    SNR means that the audio is clear.

    Forward accepts

    - ``preds``: ``shape [..., time]``
    - ``target``: ``shape [..., time]``

    Args:
        zero_mean:
            if to zero mean target and preds or not
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Raises:
        TypeError:
            if target and preds have a different shape

    Returns:
        average snr value

    Example:
        >>> import torchapi as B
        >>> from paddlemetrics import SNR
        >>> target = B.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = B.tensor([2.5, 0.0, 2.0, 8.0])
        >>> snr = SNR()
        >>> snr_val = snr(preds, target)
        >>> snr_val
        tensor(16.1805)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.

    """
    is_differentiable = True
    sum_snr: Tensor
    total: Tensor

    def __init__(
        self,
        zero_mean: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.zero_mean = zero_mean

        self.add_state("sum_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        snr_batch = snr(preds=preds, target=target, zero_mean=self.zero_mean)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SNR."""
        return self.sum_snr / self.total
