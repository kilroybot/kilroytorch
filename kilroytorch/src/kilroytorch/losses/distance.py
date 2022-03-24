from abc import ABC, abstractmethod

from torch import Tensor, nn

from kilroytorch.utils import ShapeValidator


class DistanceLoss(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.input_validator = ShapeValidator((None, None))
        self.output_validator = ShapeValidator(())

    def calculate(self, x: Tensor, y: Tensor) -> Tensor:
        self.input_validator.validate(x)
        self.input_validator.validate(y)
        out = self.calculate_internal(x, y)
        self.output_validator.validate(out)
        return out

    @abstractmethod
    def calculate_internal(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.calculate(x, y)


class MeanSquaredErrorLoss(DistanceLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.MSELoss(*args, **kwargs)

    def calculate_internal(self, x: Tensor, y: Tensor) -> Tensor:
        return self.loss(x, y)
