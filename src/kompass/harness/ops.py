from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class OpTest:
    correct: bool
    max_abs_diff: float


class Op(Protocol):
    @staticmethod
    def reference(*tensors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def compare(pred: torch.Tensor, ref: torch.Tensor) -> OpTest:
        raise NotImplementedError

    @staticmethod
    def make_inputs(
        shape: tuple[int, ...],
        dtype: torch.dtype,
        requires_grad: bool,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def bytes_moved(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        raise NotImplementedError

    @staticmethod
    def flops(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        raise NotImplementedError
