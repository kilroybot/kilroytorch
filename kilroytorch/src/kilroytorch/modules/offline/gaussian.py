from typing import List, Sequence, Tuple

import torch
from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from kilroytorch.losses.distribution import (
    DistributionLoss,
    GaussianNegativeLogLikelihoodLoss,
)
from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.modules.offline.base import SimpleOfflineModule


class SimpleGaussianOfflineModule(
    SimpleOfflineModule[V, Tensor, Tensor, Tuple[Tensor, Tensor]]
):
    def __init__(
        self,
        model: DistributionModel[Tensor, Tensor],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: DistributionLoss[
            Tuple[Tensor, Tensor]
        ] = GaussianNegativeLogLikelihoodLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__(model, codec, optimizer, loss, lr_schedulers)

    def prepare_input_for_model(self, samples: List[Tensor]) -> Tensor:
        return torch.ones(1, 1)

    def prepare_model_output_for_loss(
        self, output: Tensor, samples: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        mu, sigma = output[0]
        return mu.repeat(len(samples), 1), sigma.repeat(len(samples), 1)

    def prepare_samples_for_loss(self, samples: List[Tensor]) -> Tensor:
        return torch.vstack(samples)
