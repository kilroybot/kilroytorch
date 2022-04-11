from abc import ABC
from typing import Generic, Optional, Sequence, TypeVar

from torch import Tensor, nn

from kilroytorch.models.reward.base import RewardModel
from kilroytorch.utils import ShapeValidator, build_nonlinear_layers

A = TypeVar("A", bound=Tensor)


class PlainRewardModel(RewardModel[A], ABC, Generic[A]):
    def prepare_input_for_validation(self, x: A) -> Tensor:
        return x


class DiscreteRewardModel(PlainRewardModel[Tensor]):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (16, 16),
    ) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self.net = nn.Sequential(
            nn.Flatten(0),
            nn.Embedding(n_classes, embedding_dim),
            *build_nonlinear_layers([embedding_dim] + list(hidden_dims)),
            nn.Linear(hidden_dims[-1], 1),
        )

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(x)


class ContinuousRewardModel(PlainRewardModel[Tensor]):
    def __init__(self, hidden_dims: Sequence[int] = (16, 16)) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self.net = nn.Sequential(
            *build_nonlinear_layers([1] + list(hidden_dims)),
            nn.Linear(hidden_dims[-1], 1),
        )

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(x)
