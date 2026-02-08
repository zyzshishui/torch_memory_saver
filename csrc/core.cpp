#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

#if TMS_ROCM_LEGACY_CHUNKED
// ROCm 6.x chunked allocation helpers
namespace {

int get_global_device_id(CUdevice local_device_id) {
    const char* hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
    
    if (hip_visible && strlen(hip_visible) > 0) {
        std::string devices_str(hip_visible);
        std::stringstream ss(devices_str);
        std::string device_str;
        std::vector<int> device_list;
        
        while (std::getline(ss, device_str, ',')) {
            if (!device_str.empty()) {
                device_list.push_back(std::atoi(device_str.c_str()));
            }
        }
        
        if (static_cast<size_t>(local_device_id) < device_list.size()) {
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] HIP_VISIBLE_DEVICES=" << hip_visible 
                      << " local=" << local_device_id 
                      << " -> global=" << device_list[local_device_id] << std::endl;
#endif
            return device_list[local_device_id];
        }
    }
    return local_device_id;
}

void cu_mem_create_and_map(
    CUdevice device,
    size_t aligned_size,
    void *d_mem,
    std::vector<CUmemGenericAllocationHandle>& allocHandles,
    std::vector<size_t>& chunk_sizes
) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;

    size_t num_chunks = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE;
    allocHandles.resize(num_chunks);
    chunk_sizes.resize(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        chunk_sizes[i] = MIN(aligned_size - i * MEMCREATE_CHUNK_SIZE, MEMCREATE_CHUNK_SIZE);
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        CURESULT_CHECK(hipMemCreate(&allocHandles[i], chunk_sizes[i], &prop, 0));
    }

    size_t allocated_size = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        void *map_addr = (void*)((uintptr_t)d_mem + allocated_size);
        CURESULT_CHECK(hipMemMap((hipDeviceptr_t)map_addr, chunk_sizes[i], 0, allocHandles[i], 0));
        allocated_size += chunk_sizes[i];
    }

    CUDAUtils::cu_mem_set_access(d_mem, aligned_size, device);
}

void cu_mem_unmap_and_release(
    size_t aligned_size,
    void *d_mem,
    const std::vector<CUmemGenericAllocationHandle>& allocHandles,
    const std::vector<size_t>& chunk_sizes
) {
    size_t deallocated_size = 0;
    for (size_t i = 0; i < allocHandles.size(); ++i) {
        void *map_addr = (void*)((uintptr_t)d_mem + deallocated_size);
        CURESULT_CHECK(hipMemUnmap((hipDeviceptr_t)map_addr, chunk_sizes[i]));
        deallocated_size += chunk_sizes[i];
    }

    for (size_t i = 0; i < allocHandles.size(); ++i) {
        CURESULT_CHECK(hipMemRelease(allocHandles[i]));
    }
}

} // anonymous namespace
#endif // TMS_ROCM_LEGACY_CHUNKED

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.allocFlags.compressionType = 0x0;

    size_t granularity;
    CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;
    aligned_size = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE * MEMCREATE_CHUNK_SIZE;

    int global_device_id = get_global_device_id(device);
    uint64_t node_id = (global_device_id > 3) ? 1 : 0;

    hipDeviceptr_t d_mem;
    CURESULT_CHECK(hipMemAddressReserve(&d_mem, aligned_size, granularity, 0, node_id));
    *ptr = (void*)d_mem;

    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
    cu_mem_create_and_map(device, aligned_size, *ptr, allocHandles, chunk_sizes);

    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr,
                               aligned_size, std::move(allocHandles), std::move(chunk_sizes)}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc (ROCm 6.x chunked)"
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " aligned_size=" << aligned_size << " tag=" << tag
              << std::endl;
#endif

#else
    // CUDA and ROCm 7.0+: Single allocation
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) *ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif // TMS_ROCM_LEGACY_CHUNKED
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = std::move(allocation_metadata_[ptr]);
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

#if TMS_ROCM_LEGACY_CHUNKED
    cu_mem_unmap_and_release(metadata.aligned_size, ptr, metadata.allocHandles, metadata.chunk_sizes);
    CURESULT_CHECK(hipMemAddressFree((hipDeviceptr_t)ptr, metadata.aligned_size));
#else
    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));
#endif

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " tag=" << metadata.tag
              << std::endl;
#endif

    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
#if TMS_ROCM_LEGACY_CHUNKED
            size_t backup_size = metadata.aligned_size;
#else
            size_t backup_size = metadata.size;
#endif
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, backup_size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, backup_size, cudaMemcpyDeviceToHost));
        }

#if TMS_ROCM_LEGACY_CHUNKED
        cu_mem_unmap_and_release(metadata.aligned_size, ptr, metadata.allocHandles, metadata.chunk_sizes);
#else
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
#endif

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
}

void TorchMemorySaver::resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

#if TMS_ROCM_LEGACY_CHUNKED
        cu_mem_create_and_map(metadata.device, metadata.aligned_size, ptr,
                              metadata.allocHandles, metadata.chunk_sizes);
#else
        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
#endif

        if (metadata.enable_cpu_backup) {
#if TMS_ROCM_LEGACY_CHUNKED
            size_t backup_size = metadata.aligned_size;
#else
            size_t backup_size = metadata.size;
#endif
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, backup_size, cudaMemcpyHostToDevice));

            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
#if !TMS_ROCM_LEGACY_CHUNKED
        metadata.allocHandle = newAllocHandle;
#endif
    }
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}
