from abc import abstractmethod
from typing import Collection, Dict, Generic, List, Optional, Sequence

from kilroyshare import OfflineModule
from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from kilroytorch.losses.distribution import DistributionLoss, P
from kilroytorch.models.distribution.base import A, B, DistributionModel
from kilroytorch.modules.base import BaseModule


class BaseOfflineModule(BaseModule, OfflineModule[V]):
    def __init__(
        self,
        codec: Codec[Tensor, V],
        optimizers: Collection[Optimizer],
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__(optimizers, lr_schedulers)
        self.codec = codec

    @abstractmethod
    def fit_decoded(self, samples: List[Tensor]) -> Optional[Dict[str, float]]:
        pass

    def fit(self, samples: Collection[V]) -> Optional[Dict[str, float]]:
        decoded = [self.codec.decode(sample).view(-1, 1) for sample in samples]
        return self.fit_decoded(decoded)


class SimpleOfflineModule(BaseOfflineModule[V], Generic[V, A, B, P]):
    def __init__(
        self,
        model: DistributionModel[A, B],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: DistributionLoss[P],
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__(codec, (optimizer,), lr_schedulers)
        self.model = model
        self.loss = loss

    @abstractmethod
    def prepare_input_for_model(self, samples: List[Tensor]) -> A:
        pass

    @abstractmethod
    def prepare_model_output_for_loss(
        self, output: B, samples: List[Tensor]
    ) -> P:
        pass

    @abstractmethod
    def prepare_samples_for_loss(self, samples: List[Tensor]) -> Tensor:
        pass

    @staticmethod
    def prepare_metrics(loss: Tensor) -> Dict[str, float]:
        return {"loss": loss.item()}

    def fit_decoded(self, samples: List[Tensor]) -> Optional[Dict[str, float]]:
        output = self.model(self.prepare_input_for_model(samples))
        loss = self.loss(
            self.prepare_model_output_for_loss(output, samples),
            self.prepare_samples_for_loss(samples),
        )
        self.report(loss)
        return self.prepare_metrics(loss)
