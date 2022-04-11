from abc import ABC
from typing import Generic, Optional, TypeVar

import torch
from torch import Tensor, flatten, log_softmax, nn
from torch.nn.functional import softplus
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.utils import ShapeValidator, squash_packed

A = TypeVar("A", bound=PackedSequence)
B = TypeVar("B", bound=PackedSequence)


class SequentialDistributionModel(DistributionModel[A, B], ABC, Generic[A, B]):
    def prepare_input_for_validation(self, x: A) -> Tensor:
        return x.data

    def prepare_output_for_validation(self, x: B) -> Tensor:
        return x.data


class CategoricalSequentialDistributionModel(
    SequentialDistributionModel[PackedSequence, PackedSequence]
):
    def __init__(
        self, n_classes: int, embedding_dim: int = 16, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self._output_validator = ShapeValidator((None, n_classes))
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def output_validator(self) -> Optional[ShapeValidator]:
        return self._output_validator

    def forward_internal(self, x: PackedSequence) -> PackedSequence:
        x = squash_packed(squash_packed(x, flatten), self.embedding)
        y, _ = self.lstm(x)
        y = squash_packed(y, self.linear)
        return squash_packed(y, lambda yf: log_softmax(yf, dim=-1))


class GaussianSequentialDistributionModel(
    SequentialDistributionModel[PackedSequence, PackedSequence]
):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self._output_validator = ShapeValidator((None, 2))
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def output_validator(self) -> Optional[ShapeValidator]:
        return self._output_validator

    def forward_internal(self, x: PackedSequence) -> PackedSequence:
        y, _ = self.lstm(x)
        y = squash_packed(y, self.linear)
        return squash_packed(
            y, lambda yf: torch.hstack((yf[:, [0]], softplus(yf[:, [1]])))
        )
