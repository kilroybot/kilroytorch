from abc import ABC
from typing import Generic, Optional, TypeVar

from torch import Tensor, flatten, nn
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.reward.base import RewardModel
from kilroytorch.utils import ShapeValidator, squash_packed

A = TypeVar("A", bound=PackedSequence)


class SequentialRewardModel(RewardModel[A], ABC, Generic[A]):
    def prepare_input_for_validation(self, x: A) -> Tensor:
        return x.data


class DiscreteSequentialRewardModel(SequentialRewardModel[PackedSequence]):
    def __init__(
        self, n_classes: int, embedding_dim: int = 16, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def forward_internal(self, x: PackedSequence) -> Tensor:
        x = squash_packed(squash_packed(x, flatten), self.embedding)
        _, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])


class ContinuousSequentialRewardModel(SequentialRewardModel[PackedSequence]):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def input_validator(self) -> Optional[ShapeValidator]:
        return self.i_nput_validator

    def forward_internal(self, x: PackedSequence) -> Tensor:
        _, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])
