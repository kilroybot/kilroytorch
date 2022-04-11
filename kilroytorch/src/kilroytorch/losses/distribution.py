from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import Tensor, nn

from kilroytorch.utils import ShapeValidator

P = TypeVar("P")
T = TypeVar("T")


class DistributionLoss(ABC, Generic[P, T]):
    def __init__(self) -> None:
        super().__init__()
        self.target_validator = ShapeValidator((None, None))
        self.output_validator = ShapeValidator(())

    @abstractmethod
    def validate_params(self, params: P) -> None:
        pass

    def calculate(self, params: P, target: T) -> Tensor:
        self.validate_params(params)
        self.target_validator.validate(target)
        out = self.calculate_internal(params, target)
        self.output_validator.validate(out)
        return out

    @abstractmethod
    def calculate_internal(self, params: P, target: T) -> Tensor:
        pass

    def __call__(self, params: P, target: T) -> Tensor:
        return self.calculate(params, target)


class CategoricalDistributionLoss(DistributionLoss[Tensor, Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)


class NegativeLogLikelihoodLoss(CategoricalDistributionLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(*args, **kwargs)

    def calculate_internal(self, params: Tensor, target: Tensor) -> Tensor:
        logprobs = params
        return self.loss(logprobs, target.flatten())


class GaussianDistributionLoss(DistributionLoss[Tensor, Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None, 2))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)


class GaussianNegativeLogLikelihoodLoss(GaussianDistributionLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.GaussianNLLLoss(*args, **kwargs)

    def calculate_internal(self, params: Tensor, target: Tensor) -> Tensor:
        mu, sigma = params[:, [0]], params[:, [1]]
        return self.loss(mu, target, sigma**2)
