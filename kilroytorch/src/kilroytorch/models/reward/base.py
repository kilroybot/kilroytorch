from abc import ABC
from typing import Generic

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.base import A, BaseModel, ForwardValidation


class RewardModel(BaseModel[A, Tensor], ABC, Generic[A]):
    pass


class TensorBasedRewardModel(
    ForwardValidation[Tensor, Tensor], RewardModel[Tensor], ABC
):
    def prepare_input_for_validation(self, x: Tensor) -> Tensor:
        return x

    def prepare_output_for_validation(self, x: Tensor) -> Tensor:
        return x


class PackedSequenceBasedRewardModel(
    ForwardValidation[PackedSequence, PackedSequence],
    RewardModel[PackedSequence],
    ABC,
):
    def prepare_input_for_validation(self, x: PackedSequence) -> Tensor:
        return x.data

    def prepare_output_for_validation(self, x: PackedSequence) -> Tensor:
        return x.data
