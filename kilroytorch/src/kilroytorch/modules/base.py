from typing import Collection, Optional, Sequence

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseModule:
    def __init__(
        self,
        optimizers: Collection[Optimizer],
        lr_schedulers: Sequence[_LRScheduler] = (),
    ) -> None:
        super().__init__()
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.losses = {}

    def report(self, loss: Tensor, name: Optional[str] = None) -> None:
        name = name or "__loss__"
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(loss)

    @staticmethod
    def reduce(losses: Sequence[Tensor]) -> Tensor:
        return sum(losses) / len(losses)

    def step(self) -> "BaseModule":
        for losses in self.losses.values():
            self.reduce(losses).backward()
        self.losses = {}
        for optimizer in self.optimizers:
            optimizer.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        return self
