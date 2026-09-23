import math
from pathlib import Path

import torch

from ...harness.loaders import CudaLoader, TorchLoader, TritonLoader
from ...harness.ops import OpTest
from ...harness.spec import KernelSpec


class VecAdd:
    @staticmethod
    def reference(*tensors: torch.Tensor) -> torch.Tensor:
        A, B = tensors
        return torch.add(A, B)

    @staticmethod
    def compare(pred: torch.Tensor, ref: torch.Tensor) -> OpTest:
        return OpTest(
            correct=torch.equal(pred, ref),
            max_abs_diff=torch.max(torch.abs(ref - pred)).item(),
        )

    @staticmethod
    def make_inputs(
        shape: tuple[int, ...],
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

    @staticmethod
    def bytes_moved(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return 3 * math.prod(shape) * dtype.itemsize

    @staticmethod
    def flops(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return math.prod(shape)


torch_vec_add = KernelSpec(
    op=VecAdd(),
    loader=TorchLoader(fn=torch.add),
)
vec_add_cuda_v1 = KernelSpec(
    op=VecAdd(),
    loader=CudaLoader(
        module_name="vec_add_cuda_v1",
        fn_name="vec_add_cuda_v1",
        sources=[str(Path(__file__).parent / "vecAdd.cu")],
        supports_backward=False,
    ),
)
vec_add_triton_v1 = KernelSpec(
    op=VecAdd(),
    loader=TritonLoader(
        module_path=f"{__package__}.vec_add",
        fn_name="vec_add_triton_v1",
        supports_backward=False,
    ),
)

REGISTRY: dict[str, KernelSpec] = {
    "torch_vec_add": torch_vec_add,
    "vec_add_cuda_v1": vec_add_cuda_v1,
    "vec_add_triton_v1": vec_add_triton_v1,
}
