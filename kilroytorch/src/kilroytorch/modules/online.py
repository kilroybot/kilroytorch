from abc import ABC, abstractmethod
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from kilroyshare import OnlineModule
from kilroyshare.codec import Codec
from kilroyshare.modules import V
from torch import Tensor

# noinspection PyProtectedMember,PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from kilroytorch.adapters import DataAdapter, G
from kilroytorch.generators import Generator
from kilroytorch.losses.blackbox import BlackboxLoss, ReinforceLoss
from kilroytorch.losses.distance import DistanceLoss, MeanSquaredErrorLoss
from kilroytorch.models.distribution.base import A, B, DistributionModel
from kilroytorch.models.reward.base import RewardModel
from kilroytorch.modules.base import BaseModule
from kilroytorch.utils import generate_id


class BaseOnlineModule(BaseModule, OnlineModule[str, V], ABC, Generic[V]):
    def __init__(
        self,
        adapter: DataAdapter[Tensor, A, B, Any, Any, Any],
        codec: Codec[Tensor, V],
        optimizers: Collection[Optimizer],
        lr_schedulers: Sequence[_LRScheduler] = (),
        cache: Optional[MutableMapping[str, Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__(optimizers, lr_schedulers)
        self.adapter = adapter
        self.codec = codec
        self.cache = cache or {}

    @abstractmethod
    def generate_samples(self, n: int) -> Iterator[Tuple[Tensor, Tensor]]:
        pass

    def sample(self, n: int = 1) -> Iterator[Tuple[str, V]]:
        for sample, logprob in self.generate_samples(n):
            key = generate_id()
            value = self.codec.encode(sample)
            self.cache[key] = (sample, logprob)
            yield key, value

    @abstractmethod
    def fit_internal(
        self, samples: List[Tensor], logprobs: Tensor, scores: Tensor
    ) -> Optional[Dict[str, float]]:
        pass

    def fit(self, scores: Dict[str, float]) -> Optional[Dict[str, float]]:
        samples, logprobs = [], []
        for key in scores:
            sample, logprob = self.cache.pop(key)
            samples.append(sample)
            logprobs.append(logprob)
        scores = torch.tensor(list(scores.values())).float().view(-1, 1)
        ordered = list(self.adapter.order(zip(samples, logprobs, scores)))
        return self.fit_internal(
            [sample for sample, _, _ in ordered],
            torch.vstack([logprob for _, logprob, _ in ordered]),
            torch.vstack([score for _, _, score in ordered]),
        )


class BasicOnlineModule(BaseOnlineModule[V], Generic[V, A, B, G]):
    def __init__(
        self,
        model: DistributionModel[A, B],
        generator: Generator[A, B, G],
        adapter: DataAdapter[Tensor, A, B, G, Any, Any],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: BlackboxLoss = ReinforceLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
        cache: Optional[MutableMapping[str, Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__(adapter, codec, (optimizer,), lr_schedulers, cache)
        self.model = model
        self.generator = generator
        self.loss = loss

    def generate_samples(self, n: int) -> Iterator[Tuple[Tensor, Tensor]]:
        samples, logprobs = self.generator.generate(self.model, n)
        return zip(self.adapter.iterate_generated(samples), logprobs)

    @staticmethod
    def prepare_metrics(loss: Tensor) -> Dict[str, float]:
        return {"loss": loss.item()}

    def fit_internal(
        self, samples: List[Tensor], logprobs: Tensor, scores: Tensor
    ) -> Optional[Dict[str, float]]:
        loss = self.loss(logprobs, scores)
        self.report(loss)
        return self.prepare_metrics(loss)


class ActorCriticOnlineModule(BaseOnlineModule[V], Generic[V, A, B, G]):
    def __init__(
        self,
        actor: DistributionModel[A, B],
        critic: RewardModel[G],
        generator: Generator[A, B, G],
        adapter: DataAdapter[Tensor, A, B, G, Any, Any],
        codec: Codec[Tensor, V],
        optimizers: Collection[Optimizer],
        actor_loss: BlackboxLoss = ReinforceLoss(),
        critic_loss: DistanceLoss = MeanSquaredErrorLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
        critic_codec: Optional[Codec[Tensor, V]] = None,
        cache: Optional[MutableMapping[str, Tuple[B, Tensor]]] = None,
        actor_iterations: int = 100,
    ) -> None:
        super().__init__(adapter, codec, optimizers, lr_schedulers, cache)
        self.actor = actor
        self.critic = critic
        self.generator = generator
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.critic_codec = critic_codec
        self.actor_iterations = actor_iterations

    def generate_samples(self, n: int) -> Iterator[Tuple[Tensor, Tensor]]:
        samples, logprobs = self.generator.generate(self.actor, n)
        return zip(self.adapter.iterate_generated(samples), logprobs)

    @staticmethod
    def prepare_metrics(
        critic_loss: Tensor, actor_loss: Tensor
    ) -> Dict[str, float]:
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def recode(self, samples: G) -> G:
        if self.critic_codec is None:
            return samples
        samples = self.adapter.generated_to_codec(samples)
        samples = [self.codec.encode(sample) for sample in samples]
        samples = [self.critic_codec.decode(sample) for sample in samples]
        return self.adapter.iterable_to_generated(samples)

    def fit_internal(
        self, samples: List[Tensor], logprobs: Tensor, scores: Tensor
    ) -> Optional[Dict[str, float]]:
        n_samples = len(samples)
        samples = self.recode(self.adapter.iterable_to_generated(samples))
        critic_scores = self.critic(samples)
        loss = self.critic_loss(scores, critic_scores)
        self.report(loss, "critic")
        for _ in range(self.actor_iterations):
            samples, logprobs = self.generator.generate(self.actor, n_samples)
            samples = self.recode(samples)
            scores = self.critic(samples)
            loss = self.actor_loss(logprobs, scores)
            self.report(loss, "actor")
        return self.prepare_metrics(
            self.losses["critic"][-1],
            self.reduce(self.losses["actor"][-self.actor_iterations :]),
        )
