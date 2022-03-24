from abc import abstractmethod
from typing import (
    Collection,
    Dict,
    Generic,
    Iterable,
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
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from kilroytorch.losses.blackbox import BlackboxLoss, ReinforceLoss
from kilroytorch.losses.distance import DistanceLoss, MeanSquaredErrorLoss
from kilroytorch.models.distribution.base import A, B, DistributionModel
from kilroytorch.models.reward.base import RewardModel
from kilroytorch.modules.base import BaseModule
from kilroytorch.utils import generate_id


class BaseOnlineModule(BaseModule, OnlineModule[str, V]):
    def __init__(
        self,
        codec: Codec[Tensor, V],
        optimizers: Collection[Optimizer],
        lr_schedulers: Sequence[_LRScheduler] = (),
        cache: Optional[MutableMapping[str, Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        super().__init__(optimizers, lr_schedulers)
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
    def order(
        self, zipped: Iterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor]]:
        pass

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
        ordered = list(self.order(zip(samples, logprobs, scores)))
        return self.fit_internal(
            [sample for sample, _, _ in ordered],
            torch.vstack([logprob for _, logprob, _ in ordered]),
            torch.vstack([score for _, _, score in ordered]),
        )


class SimpleOnlineModule(BaseOnlineModule[V], Generic[V, A, B]):
    def __init__(
        self,
        model: DistributionModel[A, B],
        codec: Codec[Tensor, V],
        optimizer: Optimizer,
        loss: BlackboxLoss = ReinforceLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
        cache: Optional[MutableMapping[str, Tuple[B, Tensor]]] = None,
    ) -> None:
        super().__init__(codec, (optimizer,), lr_schedulers, cache)
        self.model = model
        self.loss = loss

    @abstractmethod
    def prepare_output_for_codec(self, x: B) -> Iterable[Tensor]:
        pass

    def generate_samples(self, n: int) -> Iterator[Tuple[Tensor, Tensor]]:
        samples, logprobs = self.model.sample(n)
        return zip(self.prepare_output_for_codec(samples), logprobs)

    @staticmethod
    def prepare_metrics(loss: Tensor) -> Dict[str, float]:
        return {"loss": loss.item()}

    def fit_internal(
        self, samples: List[Tensor], logprobs: Tensor, scores: Tensor
    ) -> Optional[Dict[str, float]]:
        loss = self.loss(logprobs, scores)
        self.report(loss)
        return self.prepare_metrics(loss)


class ActorCriticOnlineModule(BaseOnlineModule[V], Generic[V, A, B]):
    def __init__(
        self,
        actor: DistributionModel[A, B],
        critic: RewardModel[B],
        codec: Codec[Tensor, V],
        optimizers: Collection[Optimizer],
        actor_loss: BlackboxLoss = ReinforceLoss(),
        critic_loss: DistanceLoss = MeanSquaredErrorLoss(),
        lr_schedulers: Sequence[_LRScheduler] = (),
        cache: Optional[MutableMapping[str, Tuple[B, Tensor]]] = None,
        actor_iterations: int = 100,
    ) -> None:
        super().__init__(codec, optimizers, lr_schedulers, cache)
        self.actor = actor
        self.critic = critic
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.actor_iterations = actor_iterations

    @abstractmethod
    def prepare_output_for_codec(self, x: B) -> Iterable[Tensor]:
        pass

    def generate_samples(self, n: int) -> Iterator[Tuple[Tensor, Tensor]]:
        samples, logprobs = self.actor.sample(n)
        return zip(self.prepare_output_for_codec(samples), logprobs)

    @abstractmethod
    def prepare_samples_for_critic(self, samples: List[Tensor]) -> B:
        pass

    @staticmethod
    def prepare_metrics(
        critic_loss: Tensor, actor_loss: Tensor
    ) -> Dict[str, float]:
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def fit_internal(
        self, samples: List[Tensor], logprobs: Tensor, scores: Tensor
    ) -> Optional[Dict[str, float]]:
        n_samples = len(samples)
        critic_scores = self.critic(self.prepare_samples_for_critic(samples))
        loss = self.critic_loss(scores, critic_scores)
        self.report(loss, "critic")
        for _ in range(self.actor_iterations):
            samples, logprobs = self.actor.sample(n_samples)
            scores = self.critic(samples)
            loss = self.actor_loss(logprobs, scores)
            self.report(loss, "actor")
        return self.prepare_metrics(
            self.losses["critic"][-1],
            self.reduce(self.losses["actor"][-self.actor_iterations :]),
        )
