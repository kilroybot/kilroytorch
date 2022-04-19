from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Tuple, TypeVar

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.models.base import A, B
from kilroytorch.models.distribution.base import DistributionModel
from kilroytorch.models.distribution.plain import A as PA, B as PB
from kilroytorch.models.distribution.sequential import A as SA, B as SB
from kilroytorch.samplers.base import P, Sampler
from kilroytorch.utils import (
    pack_list,
    pack_padded,
    unpack_to_padded,
)

G = TypeVar("G")
PG = TypeVar("PG", bound=Tensor)
SG = TypeVar("SG", bound=PackedSequence)


class Generator(ABC, Generic[A, B, G]):
    @abstractmethod
    def generate(
        self, model: DistributionModel[A, B], n: int = 1
    ) -> Tuple[G, Tensor]:
        pass


class PlainGenerator(Generator[PA, PB, PG], Generic[PA, PB, PG]):
    def __init__(self, sampler: Sampler[B]) -> None:
        super().__init__()
        self.sampler = sampler

    def generate(
        self, model: DistributionModel[PA, PB], n: int = 1
    ) -> Tuple[PG, Tensor]:
        samples, logprobs = self.sampler.sample(model(torch.ones(1, 1)), n)
        return samples[0], logprobs[0]


class SequentialGenerator(Generator[SA, SB, SG], Generic[SA, SB, P, SG]):
    @dataclass
    class State:
        current_sequences: Tensor
        current_logprobs: Tensor
        finished_sequences: List[Tensor] = field(default_factory=list)
        finished_logprobs: List[Tensor] = field(default_factory=list)
        iteration: int = 0

    def __init__(
        self,
        sampler: Sampler[P],
        max_length: int,
        start_value: float = 0,
        end_value: float = 1,
    ):
        self.sampler = sampler
        self.max_length = max_length
        self.start_value = start_value
        self.end_value = end_value

    def initial_state(self, n: int) -> State:
        return self.State(
            current_sequences=torch.tensor([[[self.start_value]]] * n),
            current_logprobs=torch.zeros(n, 1, 1),
        )

    def should_stop(self, state: State) -> bool:
        return (
            len(state.current_sequences) <= 0
            or (state.iteration + 1) >= self.max_length
        )

    @staticmethod
    def predict(
        model: DistributionModel[SA, SB], current_sequences: Tensor
    ) -> Tensor:
        predictions, _ = unpack_to_padded(
            model(pack_padded(current_sequences))
        )
        return predictions[:, -1]

    def pick(self, batched_logprobs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.sampler.sample(batched_logprobs)

    # noinspection PyShadowingBuiltins
    def get_finished_mask(self, next: Tensor) -> Tensor:
        return next.flatten() == self.end_value

    def update_state(
        self, state: State, next_values: Tensor, next_logprobs: Tensor
    ) -> State:
        sequences = torch.hstack(
            (state.current_sequences, next_values.view(-1, 1, 1))
        )
        logprobs = torch.hstack(
            (state.current_logprobs, next_logprobs.view(-1, 1, 1))
        )
        finished_mask = self.get_finished_mask(next_values)
        state.finished_sequences.extend(list(sequences[finished_mask]))
        state.finished_logprobs.extend(list(logprobs[finished_mask].sum(1)))
        state.current_sequences = sequences[~finished_mask]
        state.current_logprobs = logprobs[~finished_mask]
        state.iteration += 1
        return state

    @staticmethod
    def complete(state: State) -> Tuple[List[Tensor], List[Tensor]]:
        return (
            state.finished_sequences + list(state.current_sequences),
            state.finished_logprobs + list(state.current_logprobs.sum(1)),
        )

    @staticmethod
    def prepare_output(
        sequences: List[Tensor], logprobs: List[Tensor]
    ) -> Tuple[PackedSequence, Tensor]:
        ordered = sorted(
            zip(sequences, logprobs),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        sequences = pack_list([sequence for sequence, _ in ordered])
        logprobs = torch.stack([logprob for _, logprob in ordered])
        return sequences, logprobs

    def generate(
        self, model: DistributionModel[SA, SB], n: int = 1
    ) -> Tuple[SG, Tensor]:
        state = self.initial_state(n)
        while not self.should_stop(state):
            logprobs = self.predict(model, state.current_sequences)
            next_values, next_logprobs = self.pick(logprobs)
            state = self.update_state(state, next_values, next_logprobs)
        sequences, logprobs = self.complete(state)
        return self.prepare_output(sequences, logprobs)
