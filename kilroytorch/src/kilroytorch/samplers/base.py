from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

from torch import Tensor

from kilroytorch.utils import ShapeValidator

P = TypeVar("P", bound=Tensor)


class Sampler(ABC, Generic[P]):
    def __init__(self) -> None:
        super().__init__()
        self.samples_validator = ShapeValidator((None, None, 1))
        self.logprobs_validator = ShapeValidator((None, None, 1))

    def sample(self, params: P, n: int = 1) -> Tuple[Tensor, Tensor]:
        self.validate_params(params)
        samples, logprobs = self.sample_internal(params, n)
        self.samples_validator.validate(samples)
        self.logprobs_validator.validate(logprobs)
        return samples, logprobs

    @abstractmethod
    def validate_params(self, params: P) -> None:
        pass

    @abstractmethod
    def sample_internal(self, params: P, n: int = 1) -> Tuple[Tensor, Tensor]:
        pass
