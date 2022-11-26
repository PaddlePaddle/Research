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
from typing import Any, Callable, List, Optional, Tuple, Union

import paddleext.torchapi as B
from paddleext.torchapi import  Tensor

#from paddlemetrics.image.fid import NoTrainInceptionV3
from paddlemetrics.metric import Metric
from paddlemetrics.utilities import rank_zero_warn
from paddlemetrics.utilities.data import dim_zero_cat
from paddlemetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


class IS(Metric):
    r"""
    Calculates the Inception Score (IS) which is used to access how realistic generated images are.
    It is defined as

    .. math::
        IS = exp(\mathbb{E}_x KL(p(y | x ) || p(y)))

    where :math:`KL(p(y | x) || p(y))` is the KL divergence between the conditional distribution :math:`p(y|x)`
    and the margianl distribution :math:`p(y)`. Both the conditional and marginal distribution is calculated
    from features extracted from the images. The score is calculated on random splits of the images such that
    both a mean and standard deviation of the score are returned. The metric was originally proposed in [1].

    Using the default feature extraction (Inception v3 using the original weights from [2]), the input is
    expected to be mini-batches of 3-channel RGB images of shape (3 x H x W) with dtype uint8. All images
    will be resized to 299 x 299 which is the size of the original training data.

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install paddlemetrics[image]`` or
        ``pip install torch-fidelity``

    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.

    Args:
        feature:
            Either an str, integer or ``nn.Module``:

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``[N,d]`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        splits: integer determining how many splits the inception score calculation should be split among

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    References:
        [1] Improved Techniques for Training GANs
        Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
        https://arxiv.org/abs/1606.03498

        [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500

    Raises:
        ValueError:
            If ``feature`` is set to an ``str`` or ``int`` and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``str`` or ``int`` and not one of ['logits_unbiased', 64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``B.nn.Module``

    Example:
        >>> import torchapi as B
        >>> _ = B.manual_seed(123)
        >>> from paddlemetrics import IS
        >>> inception = IS()  # doctest: +SKIP
        >>> # generate some images
        >>> imgs = B.randint(0, 255, (100, 3, 299, 299), dtype=B.uint8)  # doctest: +SKIP
        >>> inception.update(imgs)  # doctest: +SKIP
        >>> inception.compute()  # doctest: +SKIP
        (tensor(1.0569), tensor(0.0113))

    """
    features: List

    def __init__(
        self,
        feature: Union[str, int, B.nn.Module] = "logits_unbiased",
        splits: int = 10,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        rank_zero_warn(
            "Metric `IS` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ValueError(
                    "IS metric requires that Torch-fidelity is installed."
                    "Either install as `pip install paddlemetrics[image]`"
                    " or `pip install torch-fidelity`"
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}," f" but got {feature}."
                )

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, B.nn.Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        self.splits = splits
        self.add_state("features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
        """
        features = self.inception(imgs)
        self.features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        features = dim_zero_cat(self.features)
        # random permute the features
        idx = B.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = B.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()
