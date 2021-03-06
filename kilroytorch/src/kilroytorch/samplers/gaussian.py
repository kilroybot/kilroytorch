from abc import ABC
from typing import Tuple

from torch import Tensor
from torch.distributions import Normal

from kilroytorch.samplers.base import Sampler
from kilroytorch.utils import ShapeValidator


class GaussianSampler(Sampler[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None, 2))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)


class ProportionalGaussianSampler(GaussianSampler):
    def sample_internal(
        self, params: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        mu, sigma = params[:, 0], params[:, 1]
        dist = Normal(mu, sigma, validate_args=False)
        samples = dist.sample((n, 1))
        return (
            samples.permute(2, 0, 1),
            dist.log_prob(samples).permute(2, 0, 1),
        )
