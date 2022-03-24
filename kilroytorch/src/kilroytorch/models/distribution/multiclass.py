from abc import ABC
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from kilroytorch.models.distribution.base import TensorBasedDistributionModel
from kilroytorch.samplers.multiclass import (
    MulticlassSampler,
    ProportionalMulticlassSampler,
)
from kilroytorch.utils import ShapeValidator


class MulticlassDistributionModel(TensorBasedDistributionModel, ABC):
    def __init__(
        self,
        n_classes: int,
        sampler: MulticlassSampler = ProportionalMulticlassSampler(),
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.sampler = sampler

    def sample(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        all_logprobs = self(torch.ones(1, 1))[0]
        return self.sampler.sample(all_logprobs, n)

    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, self.n_classes))


class LinearMulticlassDistributionModel(MulticlassDistributionModel):
    def __init__(
        self,
        n_classes: int,
        sampler: MulticlassSampler = ProportionalMulticlassSampler(),
    ) -> None:
        super().__init__(n_classes, sampler)
        self.net = nn.Sequential(nn.Linear(1, n_classes), nn.LogSoftmax(-1))

    def forward_internal(self, x: Tensor) -> Tensor:
        return self.net(x)
