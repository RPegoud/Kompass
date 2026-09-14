from pathlib import Path

import torch

from ...harness.loaders import CudaLoader, TritonLoader
from ...harness.spec import KernelSpec, Op


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
            size=shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=gen,
        )

        B = torch.randn(
            size=shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=gen,
        )

        return A, B


vec_add_cuda_v1 = KernelSpec(
    op=VecAdd,
    loader=CudaLoader(
        module_name="vec_add_fwd_v1",
        fn_name="vec_add_fwd",
        sources=[str(Path(__file__).parent / "vecAdd.cu")],
        supports_backward=False,
    ),
)
vec_add_triton_v1 = KernelSpec(
    op=VecAdd,
    loader=TritonLoader(
        module_path=f"{__package__}.vec_add",
        fn_name="vec_add_triton_v1",
        supports_backward=False,
    ),
)

REGISTRY: dict[str, KernelSpec] = {
    "vec_add_cuda_v1": vec_add_cuda_v1,
    "vec_add_triton_v1": vec_add_triton_v1,
}
