from abc import ABC
from typing import Tuple

from torch import Tensor
from torch.distributions import Categorical

from kilroytorch.samplers.base import Sampler
from kilroytorch.utils import ShapeValidator


class CategoricalSampler(Sampler[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)


class ProportionalCategoricalSampler(CategoricalSampler):
    def sample_internal(
        self, logprobs: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        dist = Categorical(logprobs.exp(), validate_args=False)
        samples = dist.sample((n, 1))
        return (
            samples.permute(2, 0, 1),
            dist.log_prob(samples).permute(2, 0, 1),
        )
