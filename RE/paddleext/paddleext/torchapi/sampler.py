"""
paddle operation for sampler
"""
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

import paddle

from paddle.io import Sampler, DistributedBatchSampler, SequenceSampler, RandomSampler
import math


def identity(x):
    """

    Args:
        x ():

    Returns:

    """
    return x



class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last


    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]



class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in paddle.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)



class DistributedSampler(Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
#                if not paddle.distributed.is_initialized():
#                    raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                paddle.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = paddle.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)

