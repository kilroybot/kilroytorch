from typing import List, Sequence

from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from kilroytorch.losses.distribution import (
    DistributionLoss,
    NegativeLogLikelihoodLoss,
)
from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.modules.offline.base import SimpleOfflineModule
from kilroytorch.utils import pack_list


class SimpleSequentialOfflineModule(
    SimpleOfflineModule[V, PackedSequence, PackedSequence, Tensor]
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

    @staticmethod
    def sort(samples: List[Tensor]) -> List[Tensor]:
        return sorted(samples, key=lambda s: len(s), reverse=True)

    @staticmethod
    def slice_samples(samples: List[Tensor], s: slice) -> List[Tensor]:
        return [sequence[s] for sequence in samples]

    def truncate_first(self, samples: List[Tensor]) -> List[Tensor]:
        return self.slice_samples(samples, slice(1, None))

    def truncate_last(self, samples: List[Tensor]) -> List[Tensor]:
        return self.slice_samples(samples, slice(-1))

    def prepare_input_for_model(self, samples: List[Tensor]) -> PackedSequence:
        samples = self.sort(samples)
        return pack_list(self.truncate_last(samples))

    def prepare_model_output_for_loss(
        self, output: PackedSequence, samples: List[Tensor]
    ) -> Tensor:
        return output.data

    def prepare_samples_for_loss(self, samples: List[Tensor]) -> Tensor:
        samples = self.sort(samples)
        return pack_list(self.truncate_first(samples)).data
