from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from kilroytorch.samplers.base import Sampler
from kilroytorch.samplers.multiclass import ProportionalMulticlassSampler
from kilroytorch.utils import (
    pack_list,
    pack_padded,
    unpack_to_padded,
)

Predictor = Callable[[PackedSequence], PackedSequence]


class Generator:
    @dataclass
    class State:
        current_sequences: Tensor
        current_logprobs: Tensor
        finished_sequences: List[Tensor] = field(default_factory=list)
        finished_logprobs: List[Tensor] = field(default_factory=list)
        iteration: int = 0

    def __init__(
        self,
        max_length: int,
        sampler: Sampler[Tensor] = ProportionalMulticlassSampler(),
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
    def predict(predictor: Predictor, current_sequences: Tensor) -> Tensor:
        predictions, _ = unpack_to_padded(
            predictor(pack_padded(current_sequences))
        )
        return predictions[:, -1]

    def pick(self, batched_logprobs: Tensor) -> Tuple[Tensor, Tensor]:
        samples, sample_logprobs = [], []
        for logprobs in batched_logprobs:
            sample, logprob = self.sampler.sample(logprobs)
            samples.append(sample[0])
            sample_logprobs.append(logprob[0])
        return torch.stack(samples), torch.stack(sample_logprobs)

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
        self, predictor: Predictor, n: int = 1
    ) -> Tuple[PackedSequence, Tensor]:
        state = self.initial_state(n)
        while not self.should_stop(state):
            logprobs = self.predict(predictor, state.current_sequences)
            next_values, next_logprobs = self.pick(logprobs)
            state = self.update_state(state, next_values, next_logprobs)
        sequences, logprobs = self.complete(state)
        return self.prepare_output(sequences, logprobs)
