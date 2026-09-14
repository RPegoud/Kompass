#pragma once // prevents redundant imports
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err_ = (expr);                                        \
    TORCH_CHECK(err_ == cudaSuccess,                                  \
                "CUDA error: ", cudaGetErrorString(err_),             \
                " at ", __FILE__, ":", __LINE__);                     \
  } while (0)

#define CUDA_CHECK_LAUNCH(sync)                                                \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    if (sync)                                                                  \
      CUDA_CHECK(cudaDeviceSynchronize());                                     \
  } while (0)
