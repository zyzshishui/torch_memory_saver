#pragma once

// Define platform macros and include appropriate headers
#if defined(USE_ROCM)
// Include HIP runtime headers for AMD ROCm platform
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>
/*
 * ROCm API Mapping References:
 * - CUDA Driver API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html
 * - CUDA Runtime API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
 */
// --- Error Handling Types and Constants ---
#define CUresult hipError_t
#define cudaError_t hipError_t
#define CUDA_SUCCESS hipSuccess
#define cudaSuccess hipSuccess
// --- Error Reporting Functions ---
#define cuGetErrorString hipDrvGetErrorString
#define cudaGetErrorString hipGetErrorString
// --- Memory Management Functions ---
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define cuMemUnmap hipMemUnmap
#define cuMemRelease hipMemRelease
#define cudaMallocHost hipHostMalloc
#define cudaMemcpy hipMemcpy
// --- Memory Copy Direction Constants ---
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
// --- Device and Stream Types ---
#define CUdevice hipDevice_t
#define cudaStream_t hipStream_t
// --- Memory Allocation Constants ---
// Chunk size for memory creation operations (2 MB)
#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
// --- Utility Macros ---
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// --- ROCm Version Feature Flags ---
// ROCm 6.x has hipMemCreate bug, requires chunked allocation workaround
// ROCm 7.0+ has fixed the bug, can use non-chunked allocation like CUDA
#if HIP_VERSION < 70000000
    #define TMS_ROCM_LEGACY_CHUNKED 1
#else
    #define TMS_ROCM_LEGACY_CHUNKED 0
#endif

// ============================================================================
// CUDA Platform Configuration (NVIDIA GPUs)
// ============================================================================
#elif defined(USE_CUDA)
#include <cuda_runtime_api.h>
#include <cuda.h>

// ============================================================================
// Error: No Platform Specified
// ============================================================================
#else
#error "USE_PLATFORM is not set"
#endif
