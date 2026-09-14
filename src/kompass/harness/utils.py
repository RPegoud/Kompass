import torch


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype), (
            f"Expected a valid torch dtype, got {dtype}"
        )
    return dtype
