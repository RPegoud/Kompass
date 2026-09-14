from typing import Protocol

import torch


class Op(Protocol):
    @staticmethod
    def reference(*tensors: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def make_inputs(
        shape: tuple[int],
        dtype: torch.dtype,
        requires_grad: bool,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]: ...


class VecAdd(Op):
    @staticmethod
    def reference(*tensors: torch.Tensor) -> torch.Tensor:
        A, B = tensors
        return torch.add(A, B)

    @staticmethod
    def make_inputs(
        shape: tuple[int],
        dtype: torch.dtype,
        requires_grad: bool,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device).manual_seed(seed)
        A = torch.randn(
            shape,
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        B = torch.randn(
            size=shape,
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return A, B
