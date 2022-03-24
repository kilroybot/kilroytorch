from typing import List, Sequence

import torch
from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from kilroytorch.losses.distribution import (
    DistributionLoss,
    NegativeLogLikelihoodLoss,
)
from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.modules.offline.base import SimpleOfflineModule


class SimpleMulticlassOfflineModule(
    SimpleOfflineModule[V, Tensor, Tensor, Tensor]
):
    def __init__(
        self,
        model: DistributionModel[Tensor, Tensor],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: DistributionLoss[Tensor] = NegativeLogLikelihoodLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__(model, codec, optimizer, loss, lr_schedulers)

    def prepare_input_for_model(self, samples: List[Tensor]) -> Tensor:
        return torch.ones(1, 1)

    def prepare_model_output_for_loss(
        self, output: Tensor, samples: List[Tensor]
    ) -> Tensor:
        return output.repeat(len(samples), 1)

    def prepare_samples_for_loss(self, samples: List[Tensor]) -> Tensor:
        return torch.vstack(samples)
