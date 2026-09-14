from typing import Any

import torch

from .loaders import Loader
from .ops import Op


class KernelSpec:
    def __init__(self, op: Op, loader: Loader) -> None:
        self.op = op
        self.loader = loader

    def build(self) -> None:
        self.kernel = self.loader.load()

    def run(self, *tensors: torch.Tensor) -> torch.Tensor:
        self.build()
        return self.kernel(*tensors)

    def test(
        self,
        shape: tuple[int],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Any]:
        self.build()
        tensors = self.op.make_inputs(
            shape=shape,
            dtype=dtype,
            requires_grad=self.loader.supports_backward,
            seed=seed,
            device=device,
        )
        pred = self.kernel(*tensors)
        ref = self.op.reference(*tensors)

        return {
            "correct": all(torch.isclose(pred, ref, rtol=rtol, atol=atol)),
            "max_abs_diff": torch.max(torch.abs(ref - pred)).item(),
            "dtype": str(dtype),
            "shape": shape,
            "seed": seed,
        }

    def bench(
        self, shape: tuple[int], dtype: torch.dtype, seed: int
    ) -> dict[str, Any]: ...
