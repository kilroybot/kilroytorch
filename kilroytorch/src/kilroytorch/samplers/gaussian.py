from abc import ABC
from typing import Tuple

from torch import Tensor
from torch.distributions import Normal

from kilroytorch.samplers.base import Sampler
from kilroytorch.utils import ShapeValidator


class GaussianSampler(Sampler[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((1, 2))
        self.samples_validator = ShapeValidator((None, 1))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)

    def validate_samples(self, samples: Tensor) -> None:
        self.samples_validator.validate(samples)


class ProportionalGaussianSampler(GaussianSampler):
    def sample_internal(
        self, params: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        mu, sigma = params[0]
        dist = Normal(mu, sigma)
        samples = dist.sample((n, 1))
        return samples, dist.log_prob(samples)
