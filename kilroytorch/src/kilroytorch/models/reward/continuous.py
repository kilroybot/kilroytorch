from abc import ABC
from typing import Optional, Sequence

from torch import Tensor, nn

from kilroytorch.models.reward.base import TensorBasedRewardModel
from kilroytorch.utils import ShapeValidator


class ContinuousRewardModel(TensorBasedRewardModel, ABC):
    def __init__(self, dimensionality: int = 1) -> None:
        super().__init__()
        self.dimensionality = dimensionality

    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, self.dimensionality))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))


class NonLinearContinuousRewardModel(ContinuousRewardModel):
    def __init__(
        self, dimensionality: int = 1, hidden_dims: Sequence[int] = (16, 16)
    ) -> None:
        super().__init__(dimensionality)
        dims = [dimensionality] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(x)
