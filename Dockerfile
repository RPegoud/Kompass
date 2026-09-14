FROM nvidia/cuda:13.3.0-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git curl build-essential \
    clang clangd \
    cmake ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen

# libtorch ships inside the pip `torch` wheel (lib/, include/, share/cmake/Torch).
# Point CMake at it so `find_package(Torch)` resolves without a separate download.
# Path is stable: UV_PROJECT_ENVIRONMENT is fixed above and .python-version pins 3.14.
ENV TORCH_INSTALL_PREFIX=/opt/venv/lib/python3.14/site-packages/torch
ENV CMAKE_PREFIX_PATH="${TORCH_INSTALL_PREFIX}/share/cmake"
ENV LD_LIBRARY_PATH="${TORCH_INSTALL_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Torch's Caffe2 CMake injects a default gencode list containing compute_50,
# which CUDA 13 no longer accepts ("nvcc fatal: Unsupported gpu architecture").
# Pinning the arch list avoids that. Override per-build for other targets,
# e.g. T4=7.5, A100=8.0, L4/L40S=8.9, H100=9.0.
ENV TORCH_CUDA_ARCH_LIST=8.0

# Cache JIT-compiled extensions and object files on persistent volumes.
ENV TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
ENV CCACHE_DIR=/root/.cache/ccache

# clangd (clang 21) predates CUDA 13 and cannot parse its headers as-is.
# clang's CUDA wrapper still includes texture_fetch_functions.h, which CUDA 13
# deleted along with the legacy texture-reference API, so stub it out. This is
# editor-only; nvcc never sees it. .clangd carries the matching -I and the
# second half of the workaround (-D_NV_RSQRT_SPECIFIER=).
RUN mkdir -p /opt/clangd-cuda-shim \
    && printf '#pragma once\n' > /opt/clangd-cuda-shim/texture_fetch_functions.h
