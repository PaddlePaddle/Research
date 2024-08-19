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
from typing import Tuple

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

from paddlemetrics.utilities.checks import _check_same_shape


def _tweedie_deviance_score_update(preds: Tensor, targets: Tensor, power: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Deviance Score for the given power. Checks for same shape
    of input tensors.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
        power: see :func:`tweedie_deviance_score`

    Example:
        >>> targets = B.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = B.tensor([4.0, 3.0, 2.0, 1.0])
        >>> _tweedie_deviance_score_update(preds, targets, power=2)
        (tensor(4.8333), tensor(4))
    """
    _check_same_shape(preds, targets)

    zero_tensor = B.zeros(preds.shape, device=preds.device)

    if 0 < power < 1:
        raise ValueError(f"Deviance Score is not defined for power={power}.")

    if power == 0:
        deviance_score = B.pow(targets - preds, exponent=2)
    elif power == 1:
        # Poisson distribution
        if B.any(preds <= 0) or B.any(targets < 0):
            raise ValueError(
                f"For power={power}, 'preds' has to be strictly positive and 'targets' cannot be negative."
            )

        deviance_score = 2 * (targets * B.log(targets / preds) + preds - targets)
    elif power == 2:
        # Gamma distribution
        if B.any(preds <= 0) or B.any(targets <= 0):
            raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")

        deviance_score = 2 * (B.log(preds / targets) + (targets / preds) - 1)
    else:
        if power < 0:
            if B.any(preds <= 0):
                raise ValueError(f"For power={power}, 'preds' has to be strictly positive.")
        elif 1 < power < 2:
            if B.any(preds <= 0) or B.any(targets < 0):
                raise ValueError(
                    f"For power={power}, 'targets' has to be strictly positive and 'preds' cannot be negative."
                )
        else:
            if B.any(preds <= 0) or B.any(targets <= 0):
                raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")

        term_1 = B.pow(B.max(targets, zero_tensor), 2 - power) / ((1 - power) * (2 - power))
        term_2 = targets * B.pow(preds, 1 - power) / (1 - power)
        term_3 = B.pow(preds, 2 - power) / (2 - power)
        deviance_score = 2 * (term_1 - term_2 + term_3)

    sum_deviance_score = B.sum(deviance_score)
    num_observations = B.tensor(B.numel(deviance_score), device=preds.device)

    return sum_deviance_score, num_observations


def _tweedie_deviance_score_compute(sum_deviance_score: Tensor, num_observations: Tensor) -> Tensor:
    """Computes Deviance Score.

    Args:
        sum_deviance_score: Sum of deviance scores accumalated until now.
        num_observations: Number of observations encountered until now.

    Example:
        >>> targets = B.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = B.tensor([4.0, 3.0, 2.0, 1.0])
        >>> sum_deviance_score, num_observations = _tweedie_deviance_score_update(preds, targets, power=2)
        >>> _tweedie_deviance_score_compute(sum_deviance_score, num_observations)
        tensor(1.2083)
    """

    return sum_deviance_score / num_observations


def tweedie_deviance_score(preds: Tensor, targets: Tensor, power: float = 0.0) -> Tensor:
    r"""
    Computes the `Tweedie Deviance Score`_ between targets and predictions:

    .. math::
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }power=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }power=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }power=2\\
        2 * (\frac{(max(y,0))^{2}}{(1 - power)(2 - power)} - \frac{y(\hat{y})^{1 - power}}{1 - power} + \frac{(\hat{y})
            ^{2 - power}}{2 - power}), & \text{otherwise}
        \end{cases}

    where :math:`y` is a tensor of targets values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: Predicted tensor with shape ``(N,...)``
        targets: Ground truth tensor with shape ``(N,...)``
        power:
            - power < 0 : Extreme stable distribution. (Requires: preds > 0.)
            - power = 0 : Normal distribution. (Requires: targets and preds can be any real numbers.)
            - power = 1 : Poisson distribution. (Requires: targets >= 0 and y_pred > 0.)
            - 1 < p < 2 : Compound Poisson distribution. (Requires: targets >= 0 and preds > 0.)
            - power = 2 : Gamma distribution. (Requires: targets > 0 and preds > 0.)
            - power = 3 : Inverse Gaussian distribution. (Requires: targets > 0 and preds > 0.)
            - otherwise : Positive stable distribution. (Requires: targets > 0 and preds > 0.)

    Example:
        >>> from paddlemetrics.functional import tweedie_deviance_score
        >>> targets = B.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = B.tensor([4.0, 3.0, 2.0, 1.0])
        >>> tweedie_deviance_score(preds, targets, power=2)
        tensor(1.2083)

    """
    sum_deviance_score, num_observations = _tweedie_deviance_score_update(preds, targets, power=power)
    return _tweedie_deviance_score_compute(sum_deviance_score, num_observations)
