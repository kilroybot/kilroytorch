from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

from torch import Tensor, nn

from kilroytorch.utils import ShapeValidator

P = TypeVar("P")


class DistributionLoss(ABC, Generic[P]):
    def __init__(self) -> None:
        super().__init__()
        self.target_validator = ShapeValidator((None, None))
        self.output_validator = ShapeValidator(())

    @abstractmethod
    def validate_params(self, params: P) -> None:
        pass

    def calculate(self, params: P, target: Tensor) -> Tensor:
        self.validate_params(params)
        self.target_validator.validate(target)
        out = self.calculate_internal(params, target)
        self.output_validator.validate(out)
        return out

    @abstractmethod
    def calculate_internal(self, params: P, target: Tensor) -> Tensor:
        pass

    def __call__(self, params: P, target: Tensor) -> Tensor:
        return self.calculate(params, target)


class NegativeLogLikelihoodLoss(DistributionLoss[Tensor]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(*args, **kwargs)
        self.params_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)

    def calculate_internal(self, params: Tensor, target: Tensor) -> Tensor:
        logprobs = params
        return self.loss(logprobs, target.flatten())


class GaussianNegativeLogLikelihoodLoss(
    DistributionLoss[Tuple[Tensor, Tensor]]
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.GaussianNLLLoss(*args, **kwargs)
        self.params_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tuple[Tensor, Tensor]) -> None:
        mu, sigma = params
        self.params_validator.validate(mu)
        self.params_validator.validate(sigma)

    def calculate_internal(
        self, params: Tuple[Tensor, Tensor], target: Tensor
    ) -> Tensor:
        mu, sigma = params
        return self.loss(mu, target, sigma**2)
