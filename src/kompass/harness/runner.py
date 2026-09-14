from dataclasses import dataclass

import torch

from ..kernels import REGISTRY
from .spec import KernelSpec
from .utils import resolve_dtype


@dataclass()
class Payload:
    kernel: str
    shape: tuple[int]
    dtype: str
    device: torch.device
    seed: int
    gpu: str | None = None
    atol: float = 1e-5
    rtol: float = 1e-8


def run(payload: Payload) -> dict:
    spec: KernelSpec = REGISTRY[payload.kernel]

    return spec.test(
        shape=payload.shape,
        dtype=resolve_dtype(payload.dtype),
        seed=payload.seed,
        device=payload.device,
        atol=payload.atol,
        rtol=payload.rtol,
    )
