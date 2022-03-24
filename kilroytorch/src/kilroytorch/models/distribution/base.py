from abc import ABC, abstractmethod
from typing import Generic, Tuple

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.base import A, B, BaseModel, ForwardValidation


class DistributionModel(BaseModel[A, B], ABC, Generic[A, B]):
    @abstractmethod
    def sample(self, n: int = 1) -> Tuple[B, Tensor]:
        pass


class TensorBasedDistributionModel(
    ForwardValidation[Tensor, Tensor], DistributionModel[Tensor, Tensor], ABC
):
    def prepare_input_for_validation(self, x: Tensor) -> Tensor:
        return x

    def prepare_output_for_validation(self, x: Tensor) -> Tensor:
        return x


class PackedSequenceBasedDistributionModel(
    ForwardValidation[PackedSequence, PackedSequence],
    DistributionModel[PackedSequence, PackedSequence],
    ABC,
):
    def prepare_input_for_validation(self, x: PackedSequence) -> Tensor:
        return x.data

    def prepare_output_for_validation(self, x: PackedSequence) -> Tensor:
        return x.data
