import uuid
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)


class ShapeWarning(UserWarning):
    def __init__(
        self,
        actual: Sequence[int],
        expected: Sequence[Optional[int]],
        *args: object,
    ) -> None:
        super().__init__(
            f"Shape mismatch: expected: {tuple(expected)}, actual: {tuple(actual)}",
            *args,
        )


class ShapeError(TypeError):
    def __init__(
        self,
        actual: Sequence[int],
        expected: Sequence[Optional[int]],
        *args: object,
    ) -> None:
        super().__init__(
            f"Shape mismatch: expected: {tuple(expected)}, actual: {tuple(actual)}",
            *args,
        )


class ShapeValidator:
    def __init__(
        self, shape: Sequence[Optional[int]], warn: bool = True
    ) -> None:
        self.shape = tuple(shape)
        self.warn = warn

    def is_valid(self, x: Tensor) -> bool:
        if x.dim() != len(self.shape):
            return False
        for actual, expected in zip(x.shape, self.shape):
            if expected is not None and actual != expected:
                return False
        return True

    def validate(self, x: Tensor) -> None:
        if not self.is_valid(x):
            if self.warn:
                warnings.warn(ShapeWarning(x.shape, self.shape))
            else:
                raise ShapeError(x.shape, self.shape)


def generate_id() -> str:
    return uuid.uuid4().hex


def build_nonlinear_layers(
    sizes: Sequence[int], activation: nn.Module = nn.ReLU
) -> List[nn.Module]:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation())
    return layers


def sort(samples: Iterable[Tensor]) -> List[Tensor]:
    return sorted(samples, key=lambda s: len(s), reverse=True)


def slice_samples(samples: Iterable[Tensor], s: slice) -> List[Tensor]:
    return [sequence[s] for sequence in samples]


def truncate_first(samples: Iterable[Tensor]) -> List[Tensor]:
    return slice_samples(samples, slice(1, None))


def truncate_last(samples: Iterable[Tensor]) -> List[Tensor]:
    return slice_samples(samples, slice(-1))


def pad(x: Iterable[Tensor], pad_value: float = 0) -> Tuple[Tensor, List[int]]:
    return (
        pad_sequence(list(x), batch_first=True, padding_value=pad_value),
        [len(s) for s in x],
    )


def unpad(x: Tensor, lengths: Iterable[int]) -> List[Tensor]:
    return [s[:length] for s, length in zip(x, lengths)]


def pack_padded(
    x: Tensor, lengths: Optional[Iterable[int]] = None
) -> PackedSequence:
    lengths = (
        lengths if lengths is not None else torch.tensor([x.shape[1]] * len(x))
    )
    return pack_padded_sequence(x, lengths, batch_first=True)


def pack_list(x: Iterable[Tensor]) -> PackedSequence:
    x_sorted = sorted(x, key=lambda s: len(s), reverse=True)
    return pack_padded(*pad(x_sorted))


def unpack_to_padded(
    x: PackedSequence, pad_value: float = 0
) -> Tuple[Tensor, Tensor]:
    return pad_packed_sequence(x, batch_first=True, padding_value=pad_value)


def unpack_to_list(x: PackedSequence) -> List[Tensor]:
    return unpad(*unpack_to_padded(x))


def squash_packed(x, fn):
    return PackedSequence(
        fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices
    )
