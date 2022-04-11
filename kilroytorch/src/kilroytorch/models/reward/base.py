from abc import ABC
from typing import Generic, Optional

from torch import Tensor

from kilroytorch.models.base import A, BaseModel, ForwardValidation
from kilroytorch.utils import ShapeValidator


class RewardModel(
    ForwardValidation[A, Tensor], BaseModel[A, Tensor], ABC, Generic[A]
):
    def __init__(self) -> None:
        super().__init__()
        self._output_validator = ShapeValidator((None, 1))

    def output_validator(self) -> Optional[ShapeValidator]:
        return self._output_validator

    def prepare_output_for_validation(self, x: Tensor) -> Tensor:
        return x
