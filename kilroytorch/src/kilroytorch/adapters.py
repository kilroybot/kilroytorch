from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, TypeVar

import torch
from torch import Tensor

from kilroytorch.generators import G, PG, SG
from kilroytorch.losses.distribution import P, T
from kilroytorch.models.base import A, B
from kilroytorch.models.distribution.plain import A as PA, B as PB
from kilroytorch.models.distribution.sequential import A as SA, B as SB
from kilroytorch.utils import (
    pack_list,
    sort,
    truncate_first,
    truncate_last,
    unpack_to_list,
)

S = TypeVar("S", bound=Tensor)


class DataAdapter(ABC, Generic[S, A, B, G, P, T]):
    @abstractmethod
    def decoded_to_model(self, decoded: List[S]) -> A:
        pass

    @abstractmethod
    def model_to_params(self, output: B) -> P:
        pass

    @abstractmethod
    def decoded_to_target(self, decoded: List[S]) -> T:
        pass

    @abstractmethod
    def generated_to_codec(self, sample: G) -> Iterable[S]:
        pass

    @abstractmethod
    def iterate_generated(self, sample: G) -> Iterable[Tensor]:
        pass

    @abstractmethod
    def iterable_to_generated(self, sample: Iterable[Tensor]) -> G:
        pass

    @abstractmethod
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        pass


class PlainDataAdapter(
    DataAdapter[S, PA, PB, PG, P, T], Generic[S, PA, PB, PG, P, T]
):
    def decoded_to_model(self, decoded: List[S]) -> PA:
        return torch.ones(len(decoded), 1)

    def model_to_params(self, output: PB) -> P:
        return output

    def decoded_to_target(self, decoded: List[S]) -> T:
        return torch.vstack(decoded)

    def generated_to_codec(self, sample: PG) -> Iterable[S]:
        return sample

    def iterate_generated(self, sample: PG) -> Iterable[Tensor]:
        return iter(sample)

    def iterable_to_generated(self, sample: Iterable[Tensor]) -> PG:
        return torch.vstack(list(sample))

    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return zipped


class SequentialDataAdapter(
    DataAdapter[S, SA, SB, SG, P, T], Generic[S, SA, SB, SG, P, T]
):
    def decoded_to_model(self, decoded: List[S]) -> SA:
        return pack_list(truncate_last(sort(decoded)))

    def model_to_params(self, output: SB) -> P:
        return output.data

    def decoded_to_target(self, decoded: List[S]) -> T:
        return pack_list(truncate_first(sort(decoded))).data

    def generated_to_codec(self, sample: SG) -> Iterable[S]:
        return unpack_to_list(sample)

    def iterate_generated(self, sample: SG) -> Iterable[Tensor]:
        return unpack_to_list(sample)

    def iterable_to_generated(self, sample: Iterable[Tensor]) -> SG:
        return pack_list(sample)

    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return sorted(
            zipped,
            key=lambda triple: len(triple[0]),
            reverse=True,
        )
