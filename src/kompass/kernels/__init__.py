from .vec_add import REGISTRY as _vec_add

REGISTRY = {}

for r in [_vec_add]:
    dup = REGISTRY.keys() & r.keys()
    if dup:
        raise ValueError(f"duplicate kernel names: {dup}")
    REGISTRY.update(r)
