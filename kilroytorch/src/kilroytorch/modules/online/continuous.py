from typing import Iterable, List, Tuple

import torch
from kilroyshare.modules import V
from torch import Tensor

from kilroytorch.modules.online.base import (
    ActorCriticOnlineModule,
    SimpleOnlineModule,
)


class SimpleContinuousOnlineModule(SimpleOnlineModule[V, Tensor, Tensor]):
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return zipped

    def prepare_output_for_codec(self, x: Tensor) -> Iterable[Tensor]:
        return x


class ActorCriticContinuousOnlineModule(
    ActorCriticOnlineModule[V, Tensor, Tensor]
):
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        return zipped

    def prepare_output_for_codec(self, x: Tensor) -> Iterable[Tensor]:
        return x

    def prepare_samples_for_critic(self, samples: List[Tensor]) -> Tensor:
        return torch.vstack(samples)
