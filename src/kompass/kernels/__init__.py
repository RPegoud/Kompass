from ..harness.spec import KernelSpec
from .vec_add import REGISTRY as _vec_add

REGISTRY: dict[str, KernelSpec] = {}

for r in [_vec_add]:
    dup = REGISTRY.keys() & r.keys()
    if dup:
        raise ValueError(f"duplicate kernel names: {dup}")
    REGISTRY.update(r)
