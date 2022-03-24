from abc import ABC
from typing import Optional

from torch import Tensor, flatten, nn
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.reward.base import PackedSequenceBasedRewardModel
from kilroytorch.utils import ShapeValidator, squash_packed


class SequentialRewardModel(PackedSequenceBasedRewardModel, ABC):
    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))


class LSTMSequentialRewardModel(SequentialRewardModel):
    def __init__(
        self, n_classes: int, embedding_dim: int = 16, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward_internal(self, x: PackedSequence) -> Tensor:
        x = squash_packed(x, flatten)
        x = squash_packed(x, self.embedding)
        _, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])
