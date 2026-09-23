from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import torch
from tqdm.auto import tqdm
from triton.testing import do_bench

from .harness.spec import BenchResults, KernelSpec
from .harness.utils import resolve_dtype
from .kernels import REGISTRY


@dataclass()
class Payload:
    run_id: str
    kernels: list[str]
    shapes: list[tuple[int, ...]]
    dtype: str
    device: str
    quantiles: list[float] | None
    seed: int
    gpu: str | None = None


def _measure_copy_bandwidth(
    device: torch.device,
    n_bytes: int = 256 * 1024 * 1024,
    dtype: torch.dtype = torch.float32,
) -> float:
    a = torch.empty(n_bytes // dtype.itemsize, dtype=dtype, device=device)
    b = torch.empty_like(a)
    ms: float = do_bench(lambda: b.copy_(a), return_mode="median")  # type: ignore
    gb = 2 * n_bytes / 1e9  # copy_ moves bytes twice: read + write, multiply by 2
    return gb / (ms * 1e-3)


def run(payload: Payload) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dtype = resolve_dtype(payload.dtype)
    device = torch.device(payload.device)

    cp_bandwidth = _measure_copy_bandwidth(device=device, dtype=dtype)

    meta: dict[str, Any] = {
        "run_id": payload.run_id,
        "measured_copy_bandwidth_gbps": cp_bandwidth,
        "payload": asdict(payload),
    }
    times: list[dict[str, Any]] = []

    kernels_x_shapes = product(payload.kernels, payload.shapes)
    pbar = tqdm(sorted(kernels_x_shapes, key=lambda x: x[0]))

    for kernel, shape in pbar:
        pbar.set_description(f"Benchmarking {kernel} with {shape=}")
        spec: KernelSpec = REGISTRY[kernel]
        bench_res: BenchResults = spec.bench(
            shape=shape,
            dtype=dtype,
            device=device,
            seed=payload.seed,
            run_id=payload.run_id,
        )

        times.append(asdict(bench_res))

    return meta, times
