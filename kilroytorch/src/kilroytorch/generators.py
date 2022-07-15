import random
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
        waiting_sequences: List[Tensor]
        current_sequences: List[Tensor]
        current_logprobs: List[Tensor]
        current_max_length: int
        finished_sequences: List[Tensor] = field(default_factory=list)
        finished_logprobs: List[Tensor] = field(default_factory=list)

    def __init__(
        self,
        sampler: Sampler[P],
        max_length: int,
        end_values: List[float],
        context_values: List[Tensor],
    ):
        self.sampler = sampler
        self.max_length = max_length
        self.end_values = end_values
        self.context_values = context_values

    def initial_state(self, n: int) -> State:
        sampled_context = random.choices(self.context_values, k=n)

        min_length = len(min(sampled_context, key=len))
        current, waiting = [], []

        for sequence in sampled_context:
            if len(sequence) == min_length:
                current.append(sequence)
            else:
                waiting.append(sequence)
        return self.State(
            waiting_sequences=waiting,
            current_sequences=current,
            current_logprobs=[torch.zeros(1, 1) for _ in range(len(current))],
            current_max_length=min_length,
        )

    def should_stop(self, state: State) -> bool:
        return (
            len(state.current_sequences) <= 0
            or state.current_max_length >= self.max_length
        )

    @staticmethod
    def predict(
        model: DistributionModel[SA, SB], current_sequences: List[Tensor]
    ) -> Tensor:
        predictions, _ = unpack_to_padded(model(pack_list(current_sequences)))
        return predictions[:, -1]

    def pick(
        self, batched_logprobs: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        values, logprobs = self.sampler.sample(batched_logprobs)
        return list(values), list(logprobs)

    def get_finished_mask(self, next_values: List[Tensor]) -> List[bool]:
        return [
            value.flatten()[0].item() in self.end_values
            for value in next_values
        ]

    def update_state(
        self,
        state: State,
        next_values: List[Tensor],
        next_logprobs: List[Tensor],
    ) -> State:
        sequences = [
            torch.cat((current, next))
            for current, next in zip(state.current_sequences, next_values)
        ]
        logprobs = [
            torch.add(current, next)
            for current, next in zip(state.current_logprobs, next_logprobs)
        ]

        finished_mask = self.get_finished_mask(next_values)

        state.finished_sequences.extend(
            [
                sequence
                for sequence, finished in zip(sequences, finished_mask)
                if finished
            ]
        )
        state.finished_logprobs.extend(
            [
                logprob
                for logprob, finished in zip(logprobs, finished_mask)
                if finished
            ]
        )

        new_current_sequences = [
            sequence
            for sequence, finished in zip(sequences, finished_mask)
            if not finished
        ]
        new_current_logprobs = [
            logprobs
            for logprobs, finished in zip(logprobs, finished_mask)
            if not finished
        ]
        new_current_max_length = state.current_max_length + 1
        new_waiting_sequences = []

        for sequence in state.waiting_sequences:
            if len(sequence) == new_current_max_length:
                new_current_sequences.append(sequence)
                new_current_logprobs.append(torch.zeros(1, 1))
            else:
                new_waiting_sequences.append(sequence)

        state.current_sequences = new_current_sequences
        state.current_logprobs = new_current_logprobs
        state.current_max_length = new_current_max_length
        state.waiting_sequences = new_waiting_sequences

        return state

    @staticmethod
    def complete(state: State) -> Tuple[List[Tensor], List[Tensor]]:
        return (
            state.finished_sequences + state.current_sequences,
            state.finished_logprobs + state.current_logprobs,
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
