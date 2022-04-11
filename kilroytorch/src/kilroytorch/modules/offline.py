from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Generic, List, Optional, Sequence

from kilroyshare import OfflineModule
from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor
from torch.optim import Optimizer

# noinspection PyProtectedMember,PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from kilroytorch.adapters import DataAdapter
from kilroytorch.losses.distribution import DistributionLoss, P, T
from kilroytorch.models.distribution.base import A, B, DistributionModel
from kilroytorch.modules.base import BaseModule


class BaseOfflineModule(BaseModule, OfflineModule[V], ABC, Generic[V]):
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


class BasicOfflineModule(BaseOfflineModule[V], Generic[V, A, B]):
    def __init__(
        self,
        model: DistributionModel[A, B],
        adapter: DataAdapter[Tensor, A, B, Any, P, T],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: DistributionLoss[P, T],
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__(codec, (optimizer,), lr_schedulers)
        self.model = model
        self.adapter = adapter
        self.loss = loss

    @staticmethod
    def prepare_metrics(loss: Tensor) -> Dict[str, float]:
        return {"loss": loss.item()}

    def fit_decoded(self, samples: List[Tensor]) -> Optional[Dict[str, float]]:
        output = self.model(self.adapter.decoded_to_model(samples))
        loss = self.loss(
            self.adapter.model_to_params(output),
            self.adapter.decoded_to_target(samples),
        )
        self.report(loss)
        return self.prepare_metrics(loss)
