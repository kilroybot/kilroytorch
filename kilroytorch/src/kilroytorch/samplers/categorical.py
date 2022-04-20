from abc import ABC
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical

from kilroytorch.samplers.base import Sampler
from kilroytorch.utils import ShapeValidator


class CategoricalSampler(Sampler[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params_validator = ShapeValidator((None, None))

    def validate_params(self, params: Tensor) -> None:
        self.params_validator.validate(params)

    @staticmethod
    def select_logprobs(logprobs: Tensor, samples: Tensor) -> Tensor:
        logprobs = logprobs - logprobs.logsumexp(dim=-1, keepdim=True)
        return logprobs.gather(-1, samples[..., 0])[..., None]


class ProportionalCategoricalSampler(CategoricalSampler):
    def sample_internal(
        self, logprobs: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        dist = Categorical(logits=logprobs, validate_args=False)
        samples = dist.sample((n, 1))
        return (
            samples.permute(2, 0, 1),
            dist.log_prob(samples).permute(2, 0, 1),
        )


class TopKCategoricalSampler(CategoricalSampler):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.sampler = ProportionalCategoricalSampler()

    def filter_logprobs(self, logprobs: Tensor) -> Tensor:
        top = logprobs.topk(self.k)
        top_exp = top.values.exp()
        new_logprobs = torch.full_like(logprobs, -torch.inf)
        return new_logprobs.scatter(
            -1,
            top.indices,
            (top_exp / top_exp.sum(-1).unsqueeze(-1)).log(),
        )

    def sample_internal(
        self, logprobs: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        new_logprobs = self.filter_logprobs(logprobs)
        samples, _ = self.sampler.sample(new_logprobs, n)
        return samples, self.select_logprobs(logprobs, samples)


class NucleusCategoricalSampler(CategoricalSampler):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        self.sampler = ProportionalCategoricalSampler()

    def filter_logprobs(self, logprobs: Tensor) -> Tensor:
        sorted_logprobs = logprobs.sort(descending=True)
        cumulative_probs = sorted_logprobs.values.exp().cumsum(-1)
        sorted_top_indices = cumulative_probs <= self.p
        sorted_top_indices[..., 1:] = sorted_top_indices[..., :-1].clone()
        sorted_top_indices[..., 0] = True
        top_indices = sorted_top_indices.gather(
            -1, sorted_logprobs.indices.argsort()
        )
        new_logprobs = torch.full_like(logprobs, -torch.inf)
        return new_logprobs.masked_scatter(top_indices, logprobs[top_indices])

    def sample_internal(
        self, logprobs: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        new_logprobs = self.filter_logprobs(logprobs)
        samples, _ = self.sampler.sample(new_logprobs, n)
        return samples, self.select_logprobs(logprobs, samples)


class WithEpsilon(CategoricalSampler):
    def __init__(self, epsilon: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def mutate_samples(self, n_choices: int, samples: Tensor) -> Tensor:
        mutation_mask = Bernoulli(self.epsilon).sample(samples.shape).bool()
        uniform_probs = torch.ones(n_choices)
        uniform_samples = Categorical(uniform_probs).sample(samples.shape)
        samples[mutation_mask] = uniform_samples[mutation_mask]
        return samples

    def sample_internal(
        self, logprobs: Tensor, n: int = 1
    ) -> Tuple[Tensor, Tensor]:
        samples, _ = super().sample_internal(logprobs, n)
        samples = self.mutate_samples(logprobs.shape[-1], samples)
        return samples, self.select_logprobs(logprobs, samples)


class EpsilonProportionalCategoricalSampler(
    WithEpsilon, ProportionalCategoricalSampler
):
    pass


class EpsilonTopKCategoricalSampler(WithEpsilon, TopKCategoricalSampler):
    pass


class EpsilonNucleusCategoricalSampler(WithEpsilon, NucleusCategoricalSampler):
    pass
