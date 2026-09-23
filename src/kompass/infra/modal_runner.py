import json
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from ..harness.utils import git_commit, repo_root
from ..runner import Payload

CUDA_VOLUME_NAME = "cuda_kernels"
TRITON_VOLUME_NAME = "triton_kernels"
KERNEL_REMOTE_PATH = "/kernels"
CUDA_KERNEL_CACHE_PATH = "/cache/cuda"
TRITON_KERNEL_CACHE_PATH = "/cache/triton"

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.3.0-devel-ubuntu26.04",
        add_python="3.14",
    )
    .entrypoint([])
    .env(
        {
            "TORCH_EXTENSIONS_DIR": CUDA_KERNEL_CACHE_PATH,
            "TRITON_CACHE_DIR": TRITON_KERNEL_CACHE_PATH,
        }
    )
    .uv_sync()
    .add_local_dir(
        "./src/kompass",
        copy=False,
        remote_path=KERNEL_REMOTE_PATH,
        ignore=["*.venv"],
    )
)
app = modal.App("cuda-compile-and-run", image=cuda_image)
cuda_vol = modal.Volume.from_name(CUDA_VOLUME_NAME)
triton_vol = modal.Volume.from_name(TRITON_VOLUME_NAME)


def describe_env() -> dict:
    import torch

    def _sh(*argv) -> str:
        r = subprocess.run(argv, check=False, capture_output=True, text=True)
        return (
            r.stdout.strip()
            if r.returncode == 0
            else f"FAILED({r.returncode}): {r.stderr.strip()}"
        )

    m = re.search(r"release (\d+\.\d+), V(\S+)", _sh("nvcc", "--version"))
    nvcc = m.group(2) if m else "unknown"

    return {
        "nvidia_smi": _sh(
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ),
        "nvcc_version": nvcc,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_cuda_version": torch.version.cuda,
    }


@app.function(
    image=cuda_image,
    volumes={CUDA_KERNEL_CACHE_PATH: cuda_vol, TRITON_KERNEL_CACHE_PATH: triton_vol},
)
def clear_cache() -> None:  # TODO:
    raise NotImplementedError


@app.function(
    image=cuda_image,
    volumes={CUDA_KERNEL_CACHE_PATH: cuda_vol, TRITON_KERNEL_CACHE_PATH: triton_vol},
)
def run_test(payload: Payload) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ..runner import run

    env = describe_env()
    print(env)
    payload.gpu = env["device_name"]

    return run(payload)


if __name__ == "__main__":
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    run_id = uuid.uuid4().hex[:8]

    output_dir = repo_root() / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = output_dir / f"meta_{timestamp}_{run_id}.json"
    times_file = output_dir / f"times_{timestamp}_{run_id}.jsonl"

    meta, times = {}, []

    shapes = [(2**i,) for i in range(10, 29)]
    dtype = "float32"
    seed = 0

    payload = Payload(
        run_id=run_id,
        kernels=["vec_add_cuda_v1", "vec_add_triton_v1", "torch_vec_add"],
        gpu=None,
        shapes=shapes,
        dtype=dtype,
        device="cuda:0",
        seed=seed,
        quantiles=[0.2, 0.5, 0.8],
    )

    try:
        with modal.enable_output(), app.run():
            meta, times = run_test.with_options(gpu="T4", retries=0).remote(payload)
    except Exception as e:
        meta["error"] = repr(e)
        raise
    finally:
        meta |= {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": git_commit(),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(times_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row | {"run_id": run_id}) + "\n" for row in times)
