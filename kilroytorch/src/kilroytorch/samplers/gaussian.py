from abc import ABC
from typing import Tuple

from torch import Tensor
from torch.distributions import Normal

from kilroytorch.samplers.base import Sampler
from kilroytorch.utils import ShapeValidator


class GaussianSampler(Sampler[Tuple[Tensor, Tensor]], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None,))
        self.samples_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tuple[Tensor, Tensor]) -> None:
        mu, sigma = params
        self.params_validator.validate(mu)
        self.params_validator.validate(sigma)

    def validate_samples(self, samples: Tensor) -> None:
        self.samples_validator.validate(samples)


class ProportionalGaussianSampler(GaussianSampler):
    def sample_internal(
        self, params: Tuple[Tensor, Tensor], n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        mu, sigma = params
        dist = Normal(mu, sigma)
        samples = dist.sample([n])
        return samples, dist.log_prob(samples)
