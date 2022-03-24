from abc import ABC
from typing import Optional, Tuple

from torch import Tensor, flatten, log_softmax, nn
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.generator import Generator
from kilroytorch.models.distribution.base import (
    PackedSequenceBasedDistributionModel,
)
from kilroytorch.utils import ShapeValidator, squash_packed


class SequentialDistributionModel(PackedSequenceBasedDistributionModel, ABC):
    def __init__(self, n_classes: int, generator: Generator) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.generator = generator

    def sample(self, n: int = 1) -> Tuple[PackedSequence, Tensor]:
        return self.generator.generate(self.forward, n)

    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, self.n_classes))


class LSTMSequentialDistributionModel(SequentialDistributionModel):
    def __init__(
        self,
        n_classes: int,
        generator: Generator = Generator(10),
        embedding_dim: int = 16,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__(n_classes, generator)
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward_internal(self, x: PackedSequence) -> PackedSequence:
        x = squash_packed(x, flatten)
        x = squash_packed(x, self.embedding)
        out, _ = self.lstm(x)
        y = squash_packed(out, self.linear)
        return squash_packed(y, lambda yf: log_softmax(yf, dim=-1))
