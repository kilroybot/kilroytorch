from abc import ABC
from typing import Generic, Optional, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.utils import ShapeValidator

A = TypeVar("A", bound=Tensor)
B = TypeVar("B", bound=Tensor)


class PlainDistributionModel(DistributionModel[A, B], ABC, Generic[A, B]):
    def prepare_input_for_validation(self, x: A) -> Tensor:
        return x

    def prepare_output_for_validation(self, x: B) -> Tensor:
        return x


class CategoricalDistributionModel(PlainDistributionModel[Tensor, Tensor]):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self._output_validator = ShapeValidator((None, n_classes))
        self.net = nn.Sequential(nn.Linear(1, n_classes), nn.LogSoftmax(-1))

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def output_validator(self) -> Optional[ShapeValidator]:
        return self._output_validator

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(x)


class GaussianDistributionModel(PlainDistributionModel[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self._input_validator = ShapeValidator((None, 1))
        self._output_validator = ShapeValidator((None, 2))
        self.net = nn.Linear(1, 2)

    def input_validator(self) -> Optional[ShapeValidator]:
        return self._input_validator

    def output_validator(self) -> Optional[ShapeValidator]:
        return self._output_validator

    def forward_internal(self, x: A) -> B:
        y = self.net(x)
        return torch.hstack((y[:, [0]], softplus(y[:, [1]])))
