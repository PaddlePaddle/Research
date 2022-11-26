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
from typing import Any, List, Optional

import paddleext.torchapi as B
#import torchapi.nn.functional as F
from paddleext.torchapi import Tensor


def reduce(to_reduce: Tensor, reduction: str) -> Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        to_reduce: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given
    """
    if reduction == "elementwise_mean":
        return B.mean(to_reduce)
    if reduction == "none":
        return to_reduce
    if reduction == "sum":
        return B.sum(to_reduce)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(num: Tensor, denom: Tensor, weights: Tensor, class_reduction: str = "none") -> Tensor:
    """
    Function used to reduce classification metrics of the form `num / denom * weights`.
    For example for calculating standard accuracy the num would be number of
    true positives per class, denom would be the support per class, and weights
    would be a tensor of 1s

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'`` or ``None``: returns calculated metric per class

    Raises:
        ValueError:
            If ``class_reduction`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``.

    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    if class_reduction == "micro":
        fraction = B.sum(num) / B.sum(denom)
    else:
        fraction = num / denom

    # We need to take care of instances where the denom can be 0
    # for some (or all) classes which will produce nans
    fraction[fraction != fraction] = 0

    if class_reduction == "micro":
        return fraction
    if class_reduction == "macro":
        return B.mean(fraction)
    if class_reduction == "weighted":
        return B.sum(fraction * (weights.float() / B.sum(weights)))
    if class_reduction == "none" or class_reduction is None:
        return fraction

    raise ValueError(
        f"Reduction parameter {class_reduction} unknown." f" Choose between one of these: {valid_reduction}"
    )


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [B.zeros_like(result) for _ in range(world_size)]
    B.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = B.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = B.distributed.get_world_size(group)
    B.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = B.tensor(result.shape, device=result.device)
    local_sizes = [B.zeros_like(local_size) for _ in range(world_size)]
    B.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = B.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = B.pad(result, pad_dims)
    gathered_result = [B.zeros_like(result_padded) for _ in range(world_size)]
    B.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result
