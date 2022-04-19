from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from torch import Tensor, nn

from kilroytorch.config import config
from kilroytorch.utils import ShapeValidator

A = TypeVar("A")
B = TypeVar("B")


class BaseModel(nn.Module, ABC, Generic[A, B]):
    @abstractmethod
    def forward(self, x: A) -> B:
        pass

    def __call__(self, x: A) -> B:
        return self.forward(x)


class ForwardValidation(ABC, Generic[A, B]):
    @abstractmethod
    def input_validator(self) -> Optional[ShapeValidator]:
        pass

    @abstractmethod
    def output_validator(self) -> Optional[ShapeValidator]:
        pass

    @abstractmethod
    def prepare_input_for_validation(self, x: A) -> Tensor:
        pass

    @abstractmethod
    def prepare_output_for_validation(self, x: B) -> Tensor:
        pass

    @abstractmethod
    def forward_internal(self, x: A) -> B:
        pass

    def enable_validation(self) -> Optional[bool]:
        return None

    def forward(self, x: A) -> B:
        validation_enabled = self.enable_validation()
        if validation_enabled is None:
            validation_enabled = config.enable_validation
        if validation_enabled:
            input_validator = self.input_validator()
            if input_validator is not None:
                input = self.prepare_input_for_validation(x)
                input_validator.validate(input)
        out = self.forward_internal(x)
        if validation_enabled:
            output_validator = self.output_validator()
            if output_validator is not None:
                output = self.prepare_output_for_validation(out)
                output_validator.validate(output)
        return out
