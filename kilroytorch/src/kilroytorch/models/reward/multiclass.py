from abc import ABC
from typing import Optional, Sequence

from torch import Tensor, nn
from torch.nn.functional import one_hot

from kilroytorch.models.reward.base import TensorBasedRewardModel
from kilroytorch.utils import ShapeValidator


class MulticlassRewardModel(TensorBasedRewardModel, ABC):
    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))


class NonLinearMulticlassRewardModel(MulticlassRewardModel):
    def __init__(
        self, n_classes: int, hidden_dims: Sequence[int] = (16, 16)
    ) -> None:
        super().__init__()
        self.transform = lambda x: one_hot(x.flatten(), n_classes).float()
        dims = [n_classes] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(self.transform(x))
