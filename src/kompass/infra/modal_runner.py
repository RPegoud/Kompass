import re
import subprocess

import modal
import torch

from ..harness import Payload

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
def run_test(payload: Payload) -> dict:
    from ..harness import run

    env = describe_env()
    print(env)
    payload.gpu = env["device_name"]

    return run(payload)


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        shape = (1024,)
        dtype = "float32"
        seed = 0

        print("---CUDA---")
        payload = Payload(
            kernel="vec_add_cuda_v1",
            gpu=None,
            shape=shape,
            dtype=dtype,
            device=torch.device("cuda:0"),
            seed=seed,
        )

        res = run_test.with_options(gpu="T4", retries=0).remote(payload)
        print(res)

        print("---TRITON---")
        payload = Payload(
            kernel="vec_add_triton_v1",
            gpu=None,
            shape=shape,
            dtype=dtype,
            device=torch.device("cuda:0"),
            seed=seed,
        )
        res = run_test.with_options(gpu="T4", retries=0).remote(payload)
        print(res)
