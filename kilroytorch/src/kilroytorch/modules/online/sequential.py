from typing import Iterable, List, Tuple

from kilroyshare.modules import V
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.modules.online.base import (
    ActorCriticOnlineModule,
    SimpleOnlineModule,
)
from kilroytorch.utils import pack_list, unpack_to_list


class SimpleSequentialOnlineModule(
    SimpleOnlineModule[V, PackedSequence, PackedSequence]
):
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return sorted(
            zipped,
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    def prepare_output_for_codec(self, x: PackedSequence) -> Iterable[Tensor]:
        return unpack_to_list(x)


class ActorCriticSequentialOnlineModule(
    ActorCriticOnlineModule[V, PackedSequence, PackedSequence]
):
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return sorted(
            zipped,
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    def prepare_output_for_codec(self, x: PackedSequence) -> Iterable[Tensor]:
        return unpack_to_list(x)

    def prepare_samples_for_critic(
        self, samples: List[Tensor]
    ) -> PackedSequence:
        return pack_list(samples)
