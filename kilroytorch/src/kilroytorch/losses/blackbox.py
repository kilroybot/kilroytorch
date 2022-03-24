from abc import ABC, abstractmethod

from torch import Tensor

from kilroytorch.utils import ShapeValidator


class BlackboxLoss(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.input_validator = ShapeValidator((None, 1))
        self.output_validator = ShapeValidator(())

    def calculate(self, logprobs: Tensor, scores: Tensor) -> Tensor:
        self.input_validator.validate(logprobs)
        self.input_validator.validate(scores)
        out = self.calculate_internal(logprobs, scores)
        self.output_validator.validate(out)
        return out

    @abstractmethod
    def calculate_internal(self, logprobs: Tensor, scores: Tensor) -> Tensor:
        pass

    def __call__(self, logprobs: Tensor, scores: Tensor) -> Tensor:
        return self.calculate(logprobs, scores)


class ReinforceLoss(BlackboxLoss):
    def calculate_internal(self, logprobs: Tensor, scores: Tensor) -> Tensor:
        return -(scores * logprobs).mean()
