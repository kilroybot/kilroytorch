from abc import ABC
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from kilroytorch.models.distribution.base import TensorBasedDistributionModel
from kilroytorch.samplers.gaussian import (
    GaussianSampler,
    ProportionalGaussianSampler,
)
from kilroytorch.utils import ShapeValidator


class GaussianDistributionModel(TensorBasedDistributionModel, ABC):
    def __init__(
        self, sampler: GaussianSampler = ProportionalGaussianSampler()
    ) -> None:
        super().__init__()
        self.sampler = sampler

    def sample(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        mu, sigma = self(torch.ones(1, 1))[0]
        return self.sampler.sample((mu.view(-1), sigma.view(-1)), n)

    def input_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return ShapeValidator((None, 2))


class LinearGaussianDistributionModel(GaussianDistributionModel):
    def __init__(
        self, sampler: GaussianSampler = ProportionalGaussianSampler()
    ) -> None:
        super().__init__(sampler)
        self.mu_net = nn.Linear(1, 1)
        self.sigma_net = nn.Sequential(nn.Linear(1, 1), nn.Softplus())

    def forward_internal(self, x: Tensor) -> Tensor:
        mu = self.mu_net(x)
        sigma = self.sigma_net(x)
        return torch.hstack((mu, sigma))
