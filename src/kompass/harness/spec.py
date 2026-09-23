from collections.abc import Callable
from dataclasses import dataclass

import torch
from triton.testing import do_bench

from .loaders import Loader
from .ops import Op, OpTest


@dataclass(frozen=True)
class TestResults:
    correct: bool
    max_abs_diff: float
    dtype: str
    shape: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class BenchResults:
    kernel_name: str
    shape: tuple[int, ...]
    dtype: str
    seed: int
    times: dict[str, list[float]] | list[float]
    flops: int
    bytes_moved: int
    device: str


class KernelSpec:
    def __init__(self, op: Op, loader: Loader) -> None:
        self.op = op
        self.loader = loader
        self.name = self.loader.fn_name
        self._kernel = None

    @property
    def kernel(self) -> Callable:
        if self._kernel is None:
            self._kernel = self.loader.load()
        return self._kernel

    def _check(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        seed: int,
    ) -> TestResults:
        pred = self.kernel(*tensors)
        ref = self.op.reference(*tensors)
        res: OpTest = self.op.compare(pred, ref)
        return TestResults(
            correct=res.correct,
            max_abs_diff=res.max_abs_diff,
            dtype=str(dtype),
            shape=shape,
            seed=seed,
        )

    def _make_inputs(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> tuple[torch.Tensor, ...]:
        return self.op.make_inputs(
            shape=shape,
            dtype=dtype,
            requires_grad=self.loader.supports_backward,
            seed=seed,
            device=device,
        )

    def test(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> TestResults:
        tensors = self.op.make_inputs(
            shape=shape,
            dtype=dtype,
            requires_grad=self.loader.supports_backward,
            seed=seed,
            device=device,
        )
        return self._check(tensors, shape=shape, dtype=dtype, seed=seed)

    def bench(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        run_id: str,
        warmup: int = 25,
        rep: int = 100,
        quantiles: tuple[float, ...] = (0.2, 0.5, 0.8),
    ) -> BenchResults:
        tensors = self._make_inputs(shape=shape, dtype=dtype, device=device, seed=seed)

        gate = self._check(tensors=tensors, shape=shape, dtype=dtype, seed=seed)
        if not gate.correct:
            raise ValueError(
                f"The kernel did not pass the correctness test: {gate.correct=}, {gate.max_abs_diff=}\nAborting benchmark..."
            )

        times: list[float] = do_bench(
            fn=lambda: self.kernel(*tensors),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
        )  # type: ignore

        return BenchResults(
            kernel_name=self.name,
            shape=shape,
            dtype=str(dtype),
            seed=seed,
            times=times,
            flops=self.op.flops(shape, dtype),
            bytes_moved=self.op.bytes_moved(shape, dtype),
            device=str(device),
        )
