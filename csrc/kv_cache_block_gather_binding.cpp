/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Torch adapter and mapped-host registration lifecycle for the experimental
 * kv_cache_block_gather custom operator.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/core/Dict.h>
#include <ATen/core/Formatting.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace vllm_ascend {
namespace {
struct HostMapping {
    uint64_t mapping_id;
    uintptr_t host_base;
    uint64_t size;
    void* device_base;
    int32_t device_id;
    // owned=true means this extension called aclrtHostRegister and must pair it
    // with aclrtHostUnregister.  Pinned allocators configured with
    // pinned_mem_register:True already own their registration; those mappings
    // are borrowed and must never be unregistered here.
    bool owned;
    uint64_t explicit_handle_count;
};

struct HostMappingHandle {
    int64_t handle;
    uint64_t mapping_id;
    uintptr_t requested_host_addr;
    uint64_t requested_bytes;
    uintptr_t mapped_host_base;
    uint64_t mapped_bytes;
    uintptr_t device_addr;
    bool owned;
    bool already_mapped;
    bool active;
    // Keep the allocation alive for the complete registration lease.  The
    // Python integration also retains its pool tensors, but the C++ lifecycle
    // API must be safe when used directly.
    at::Tensor tensor_owner;
    pid_t owner_pid;
    int32_t device_id;
};

struct HostRange {
    uintptr_t host_addr;
    uintptr_t aligned_addr;
    uint64_t offset;
    uint64_t register_size;
};

struct HostMappingCounters {
    uint64_t register_call_count = 0;
    uint64_t register_bytes_total = 0;
    uint64_t register_bytes_current = 0;
    uint64_t register_bytes_peak = 0;
    uint64_t unregister_call_count = 0;
    uint64_t borrowed_mapping_count = 0;
    uint64_t explicit_register_call_count = 0;
    uint64_t explicit_unregister_call_count = 0;
};

struct HostMappingRegistry {
    explicit HostMappingRegistry(pid_t process_id)
        : pid(process_id),
          next_handle((static_cast<int64_t>(process_id) & 0x7fffffffLL) << 32 | 1)
    {
    }

    pid_t pid;
    std::mutex mutex;
    std::vector<std::shared_ptr<HostMapping>> mappings;
    std::unordered_map<int64_t, HostMappingHandle> handles;
    uint64_t next_mapping_id = 1;
    int64_t next_handle;
    HostMappingCounters counters;
};

HostMappingRegistry& get_host_mapping_registry()
{
    static HostMappingRegistry* registry = new HostMappingRegistry(getpid());
    const pid_t current_pid = getpid();
    // Check before taking the mutex: a child can inherit it while another
    // parent thread held it.  ACL mappings and torch-npu allocator state are not
    // fork-safe, so reject using a registry created by the parent rather than
    // deadlocking, reusing stale aliases, or tearing down parent-owned state.
    // Fork-before-first-use remains valid because the registry is then created
    // in the worker process itself.
    TORCH_CHECK(registry->pid == current_pid,
                "kv_cache_block_gather: mapped-host registry cannot be used "
                "after fork; create/register the worker-local CPU pool after "
                "the worker process has started (registry_pid=",
                registry->pid,
                ", current_pid=",
                current_pid,
                ")");
    return *registry;
}

std::mutex& get_host_mapping_mutex()
{
    return get_host_mapping_registry().mutex;
}

std::vector<std::shared_ptr<HostMapping>>& get_host_mappings()
{
    return get_host_mapping_registry().mappings;
}

std::unordered_map<int64_t, HostMappingHandle>& get_host_mapping_handles()
{
    return get_host_mapping_registry().handles;
}

uint64_t& get_next_host_mapping_id()
{
    return get_host_mapping_registry().next_mapping_id;
}

uint64_t allocate_host_mapping_id_locked()
{
    auto& next_id = get_next_host_mapping_id();
    TORCH_CHECK(next_id > 0 &&
                    next_id < std::numeric_limits<uint64_t>::max(),
                "kv_cache_block_gather: host mapping ID space exhausted");
    return next_id++;
}

int64_t& get_next_host_mapping_handle()
{
    return get_host_mapping_registry().next_handle;
}

HostMappingCounters& get_host_mapping_counters()
{
    return get_host_mapping_registry().counters;
}

pid_t& get_host_mapping_registry_pid()
{
    return get_host_mapping_registry().pid;
}

void ensure_host_mapping_registry_process_locked()
{
    TORCH_INTERNAL_ASSERT(get_host_mapping_registry_pid() == getpid());
}

uint64_t round_up_u64(uint64_t value, uint64_t alignment)
{
    TORCH_CHECK(alignment > 0,
                "kv_cache_block_gather: host mapping alignment must be positive");
    TORCH_CHECK(value <= std::numeric_limits<uint64_t>::max() - (alignment - 1),
                "kv_cache_block_gather: host mapping size overflow");
    return (value + alignment - 1) / alignment * alignment;
}

HostRange get_host_range(void* host_ptr, uint64_t bytes)
{
    TORCH_CHECK(host_ptr != nullptr, "kv_cache_block_gather: src_pages data pointer is null");
    TORCH_CHECK(bytes > 0, "kv_cache_block_gather: src_pages must not be empty");

    // aclrtHostRegister works on OS-page ranges.  Preserve the caller's offset
    // while expanding the registration to page-aligned start/end addresses.
    const long page_size_long = sysconf(_SC_PAGESIZE);
    TORCH_CHECK(page_size_long > 0, "kv_cache_block_gather: sysconf(_SC_PAGESIZE) failed");
    const uint64_t page_size = static_cast<uint64_t>(page_size_long);
    const uintptr_t host_addr = reinterpret_cast<uintptr_t>(host_ptr);
    TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - host_addr,
                "kv_cache_block_gather: host range address overflow");
    const uintptr_t aligned_addr = host_addr / page_size * page_size;
    const uint64_t offset = static_cast<uint64_t>(host_addr - aligned_addr);
    TORCH_CHECK(bytes <= std::numeric_limits<uint64_t>::max() - offset,
                "kv_cache_block_gather: host mapping size overflow");
    const uint64_t register_size = round_up_u64(offset + bytes, page_size);
    return HostRange{host_addr, aligned_addr, offset, register_size};
}

int32_t get_current_host_mapping_device()
{
    // Reject inherited post-fork registry use before touching ACL runtime state.
    (void)get_host_mapping_registry();
    int32_t device_id = -1;
    const aclError ret = aclrtGetDevice(&device_id);
    TORCH_CHECK(ret == ACL_SUCCESS && device_id >= 0,
                "kv_cache_block_gather: aclrtGetDevice failed while managing "
                "a mapped host pool, error code: ",
                ret);
    return device_id;
}

std::shared_ptr<HostMapping> find_host_mapping_locked(uintptr_t host_addr,
                                                      uint64_t bytes,
                                                      int32_t device_id)
{
    const uintptr_t request_end = host_addr + bytes;
    TORCH_CHECK(request_end >= host_addr,
                "kv_cache_block_gather: host range address overflow");
    for (const auto& mapping : get_host_mappings()) {
        const uintptr_t mapping_end = mapping->host_base + mapping->size;
        if (host_addr >= mapping->host_base && request_end <= mapping_end) {
            TORCH_CHECK(mapping->device_id == device_id,
                        "kv_cache_block_gather: mapped host pool was registered "
                        "on NPU device ",
                        mapping->device_id,
                        " but current device is ",
                        device_id,
                        "; worker-local mappings cannot cross devices");
            return mapping;
        }
    }
    return nullptr;
}

std::shared_ptr<HostMapping> find_overlapping_host_mapping_locked(
    uintptr_t host_addr,
    uint64_t bytes)
{
    const uintptr_t request_end = host_addr + bytes;
    for (const auto& mapping : get_host_mappings()) {
        const uintptr_t mapping_end = mapping->host_base + mapping->size;
        TORCH_CHECK(mapping_end >= mapping->host_base,
                    "kv_cache_block_gather: cached host mapping address overflow");
        if (host_addr < mapping_end && mapping->host_base < request_end) {
            return mapping;
        }
    }
    return nullptr;
}

void check_no_partial_host_mapping_overlap_locked(uintptr_t host_addr,
                                                  uint64_t bytes)
{
    const auto overlap =
        find_overlapping_host_mapping_locked(host_addr, bytes);
    TORCH_CHECK(overlap == nullptr,
                "kv_cache_block_gather: requested host range partially overlaps an "
                "existing mapping; register the complete worker-local pool before "
                "registering its views (request_host_ptr=",
                reinterpret_cast<void*>(host_addr),
                ", request_bytes=",
                bytes,
                ", mapping_host_base=",
                reinterpret_cast<void*>(overlap == nullptr ? 0 : overlap->host_base),
                ", mapping_bytes=",
                overlap == nullptr ? 0 : overlap->size,
                ")");
}

bool is_host_range_mapping_cached(const HostRange& range, uint64_t bytes)
{
    const int32_t device_id = get_current_host_mapping_device();
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    return find_host_mapping_locked(range.host_addr, bytes, device_id) != nullptr;
}

std::mutex& get_host_gather_opapi_mutex()
{
    static std::mutex mutex;
    return mutex;
}

void*& get_host_gather_opapi_handler()
{
    // Deliberately never dlclose this process-wide runtime. ACL executors may
    // still reference code from the custom-op library after Python teardown.
    static void* handler = nullptr;
    return handler;
}

std::string& get_host_gather_opapi_path()
{
    static std::string path;
    return path;
}

void* get_host_gather_opapi_func_addr(const char* api_name)
{
    void* handler = get_host_gather_opapi_handler();
    if (handler != nullptr) {
        // An explicitly configured OPAPI library is authoritative.  Falling
        // back to a process-global symbol after dlsym failure would hide a
        // stale/wrong deployment until behavior diverged at runtime.
        return dlsym(handler, api_name);
    }
    return GetOpApiFuncAddr(api_name);
}

void* get_mapped_host_device_ptr(void* host_ptr, uint64_t bytes)
{
    const HostRange range = get_host_range(host_ptr, bytes);
    const int32_t device_id = get_current_host_mapping_device();
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    const auto mapping =
        find_host_mapping_locked(range.host_addr, bytes, device_id);
    TORCH_CHECK(mapping != nullptr,
                "kv_cache_block_gather: src_pages must have an active explicit "
                "host-pool registration before gather");
    return static_cast<char*>(mapping->device_base) +
           (range.host_addr - mapping->host_base);
}

int64_t register_explicit_mapped_host_range(const at::Tensor& src_pages)
{
    void* host_ptr = src_pages.data_ptr();
    const uint64_t bytes =
        static_cast<uint64_t>(src_pages.numel() * src_pages.element_size());
    const HostRange range = get_host_range(host_ptr, bytes);
    const int32_t device_id = get_current_host_mapping_device();
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    auto& handles = get_host_mapping_handles();
    auto& counters = get_host_mapping_counters();
    counters.explicit_register_call_count += 1;

    // Every call gets an independent lease.  The physical mapping is reused,
    // but one pool owner cannot invalidate another owner's opaque handle.
    auto mapping = find_host_mapping_locked(range.host_addr, bytes, device_id);
    bool already_mapped = false;
    if (mapping == nullptr) {
        check_no_partial_host_mapping_overlap_locked(range.host_addr, bytes);
        // torch-npu's opt-in pinned allocator may already have registered the
        // allocation.  CANN returns ACL_SUCCESS with a null pointer for some
        // ordinary host allocations, so both conditions are required before
        // treating this as a borrowed mapping.
        void* borrowed_device_ptr = nullptr;
        const aclError lookup_ret =
            aclrtHostGetDevicePointer(host_ptr, &borrowed_device_ptr, 0);
        already_mapped = lookup_ret == ACL_SUCCESS && borrowed_device_ptr != nullptr;

        if (already_mapped) {
            mapping = std::make_shared<HostMapping>(HostMapping{
                allocate_host_mapping_id_locked(),
                range.host_addr,
                bytes,
                borrowed_device_ptr,
                device_id,
                false,
                0});
            counters.borrowed_mapping_count += 1;
        } else {
            void* mapped_base = nullptr;
            const aclError register_ret =
                aclrtHostRegister(reinterpret_cast<void*>(range.aligned_addr),
                                  range.register_size,
                                  ACL_HOST_REGISTER_MAPPED,
                                  &mapped_base);
            TORCH_CHECK(register_ret == ACL_SUCCESS,
                        "kv_cache_block_gather: aclrtHostRegister failed, error code: ",
                        register_ret,
                        ", host_ptr=",
                        host_ptr,
                        ", register_size=",
                        range.register_size);
            mapping = std::make_shared<HostMapping>(HostMapping{
                allocate_host_mapping_id_locked(),
                range.aligned_addr,
                range.register_size,
                mapped_base,
                device_id,
                true,
                0});
            counters.register_call_count += 1;
            counters.register_bytes_total += range.register_size;
            counters.register_bytes_current += range.register_size;
            counters.register_bytes_peak =
                std::max(counters.register_bytes_peak,
                         counters.register_bytes_current);
        }
        get_host_mappings().push_back(mapping);
    } else {
        already_mapped = !mapping->owned;
    }

    auto& next_handle = get_next_host_mapping_handle();
    TORCH_CHECK(next_handle > 0 &&
                    next_handle < std::numeric_limits<int64_t>::max(),
                "kv_cache_block_gather: host mapping handle space exhausted");
    const int64_t opaque_handle = next_handle++;
    mapping->explicit_handle_count += 1;
    handles.emplace(
        opaque_handle,
        HostMappingHandle{
            opaque_handle,
            mapping->mapping_id,
            range.host_addr,
            bytes,
            mapping->host_base,
            mapping->size,
            reinterpret_cast<uintptr_t>(mapping->device_base) +
                (range.host_addr - mapping->host_base),
            mapping->owned,
            already_mapped,
            true,
            src_pages,
            getpid(),
            device_id});
    return opaque_handle;
}

std::shared_ptr<HostMapping> find_host_mapping_by_id_locked(uint64_t mapping_id)
{
    for (const auto& mapping : get_host_mappings()) {
        if (mapping->mapping_id == mapping_id) {
            return mapping;
        }
    }
    return nullptr;
}

void synchronize_before_host_mapping_release_locked()
{
    // Host mappings can be consumed from any NPU stream.  Teardown is cold-path
    // control work, so a device-wide fence is preferable to invalidating GM
    // aliases while a previously-enqueued gather is still reading them.
    const aclError sync_ret = aclrtSynchronizeDevice();
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "kv_cache_block_gather: aclrtSynchronizeDevice before host "
                "unregistration failed, error code: ",
                sync_ret);
}

aclTensor* create_acl_tensor_with_data(const at::Tensor& tensor, void* data)
{
    // Reuse the CPU tensor's shape/dtype/stride metadata, but replace its data
    // pointer with the device-visible mapped alias returned by aclrtHostRegister.
    // The custom kernel can then treat src_pages like GM without staging it into
    // NPU HBM first.
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(aclCreateTensor != nullptr, "kv_cache_block_gather: aclCreateTensor not found");
    const aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                "kv_cache_block_gather: unsupported tensor dtype ",
                c10::toString(tensor.scalar_type()));
    c10::SmallVector<int64_t, 5> storage_dims;
    for (const auto dim : tensor.sizes()) {
        storage_dims.push_back(dim);
    }
    return aclCreateTensor(tensor.sizes().data(),
                           tensor.sizes().size(),
                           acl_data_type,
                           tensor.strides().data(),
                           0,
                           ACL_FORMAT_ND,
                           storage_dims.data(),
                           storage_dims.size(),
                           data);
}

} // namespace
bool has_kv_cache_block_gather_runtime()
{
    return get_host_gather_opapi_func_addr(
               "aclnnKvCacheBlockGatherGetWorkspaceSize") != nullptr &&
           get_host_gather_opapi_func_addr("aclnnKvCacheBlockGather") != nullptr;
}

bool load_kv_cache_block_gather_runtime(const std::string& path)
{
    TORCH_CHECK(!path.empty(),
                "kv_cache_block_gather: custom-op library path is empty");
    std::lock_guard<std::mutex> guard(get_host_gather_opapi_mutex());
    auto& handler = get_host_gather_opapi_handler();
    auto& loaded_path = get_host_gather_opapi_path();
    if (handler != nullptr) {
        TORCH_CHECK(loaded_path == path,
                    "kv_cache_block_gather: runtime already loaded from ",
                    loaded_path,
                    "; refusing a second library ",
                    path);
        return true;
    }

    dlerror();
    handler = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    const char* error = dlerror();
    TORCH_CHECK(handler != nullptr,
                "kv_cache_block_gather: failed to dlopen ",
                path,
                ", error=",
                error == nullptr ? "unknown" : error);
    loaded_path = path;
    return has_kv_cache_block_gather_runtime();
}

bool is_kv_cache_block_gather_host_mapping_cached(const at::Tensor& src_pages)
{
    if (!src_pages.defined() || src_pages.numel() == 0) {
        return false;
    }
    if (!src_pages.device().is_cpu() || !src_pages.is_contiguous()) {
        return false;
    }
    const uint64_t bytes =
        static_cast<uint64_t>(src_pages.numel() * src_pages.element_size());
    const HostRange range = get_host_range(src_pages.data_ptr(), bytes);
    return is_host_range_mapping_cached(range, bytes);
}

c10::Dict<std::string, int64_t> get_kv_cache_block_gather_host_mapping_stats()
{
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    c10::Dict<std::string, int64_t> stats;
    const auto& mappings = get_host_mappings();
    const auto& counters = get_host_mapping_counters();
    uint64_t current_bytes = 0;
    uint64_t max_mapping_size = 0;
    uint64_t owned_mapping_count = 0;
    uint64_t borrowed_mapping_count = 0;
    for (const auto& mapping : mappings) {
        current_bytes += mapping->size;
        if (mapping->size > max_mapping_size) {
            max_mapping_size = mapping->size;
        }
        owned_mapping_count += mapping->owned ? 1 : 0;
        borrowed_mapping_count += mapping->owned ? 0 : 1;
    }
    stats.insert("mapping_count", static_cast<int64_t>(mappings.size()));
    stats.insert("mapped_bytes_current", static_cast<int64_t>(current_bytes));
    stats.insert("mapped_bytes_peak", static_cast<int64_t>(counters.register_bytes_peak));
    stats.insert("mapped_bytes_max_region", static_cast<int64_t>(max_mapping_size));
    stats.insert("register_call_count", static_cast<int64_t>(counters.register_call_count));
    stats.insert("register_bytes_total", static_cast<int64_t>(counters.register_bytes_total));
    stats.insert("unregister_call_count", static_cast<int64_t>(counters.unregister_call_count));
    stats.insert("owned_mapping_count", static_cast<int64_t>(owned_mapping_count));
    stats.insert("borrowed_mapping_count", static_cast<int64_t>(borrowed_mapping_count));
    stats.insert("borrowed_mapping_count_total",
                 static_cast<int64_t>(counters.borrowed_mapping_count));
    stats.insert("explicit_handle_count",
                 static_cast<int64_t>(std::count_if(
                     get_host_mapping_handles().begin(),
                     get_host_mapping_handles().end(),
                     [](const auto& item) { return item.second.active; })));
    stats.insert("explicit_handle_storage_count",
                 static_cast<int64_t>(get_host_mapping_handles().size()));
    stats.insert("explicit_register_call_count",
                 static_cast<int64_t>(counters.explicit_register_call_count));
    stats.insert("explicit_unregister_call_count",
                 static_cast<int64_t>(counters.explicit_unregister_call_count));
    stats.insert("registry_pid",
                 static_cast<int64_t>(get_host_mapping_registry_pid()));
    return stats;
}

int64_t register_kv_cache_block_gather_host_pool(const at::Tensor& src_pages)
{
    TORCH_CHECK(src_pages.defined(),
                "kv_cache_block_gather: src_pages is undefined");
    TORCH_CHECK(src_pages.numel() > 0,
                "kv_cache_block_gather: src_pages must not be empty");
    TORCH_CHECK(src_pages.device().is_cpu(),
                "kv_cache_block_gather: src_pages must be a CPU tensor");
    TORCH_CHECK(src_pages.is_contiguous(),
                "kv_cache_block_gather: src_pages must be contiguous");
    return register_explicit_mapped_host_range(src_pages);
}

c10::Dict<std::string, int64_t> inspect_kv_cache_block_gather_host_pool(
    int64_t opaque_handle)
{
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    c10::Dict<std::string, int64_t> info;
    info.insert("handle", opaque_handle);
    const auto& handles = get_host_mapping_handles();
    const auto it = handles.find(opaque_handle);
    if (it == handles.end()) {
        info.insert("active", 0);
        info.insert("known", 0);
        return info;
    }

    const auto& handle = it->second;
    info.insert("known", 1);
    info.insert("active", handle.active ? 1 : 0);
    info.insert("owned", handle.owned ? 1 : 0);
    info.insert("already_mapped", handle.already_mapped ? 1 : 0);
    info.insert("mapping_id", static_cast<int64_t>(handle.mapping_id));
    info.insert("host_ptr", static_cast<int64_t>(handle.requested_host_addr));
    info.insert("device_ptr", static_cast<int64_t>(handle.device_addr));
    info.insert("requested_bytes", static_cast<int64_t>(handle.requested_bytes));
    info.insert("mapped_host_base", static_cast<int64_t>(handle.mapped_host_base));
    info.insert("mapped_bytes", static_cast<int64_t>(handle.mapped_bytes));
    info.insert("owner_pid", static_cast<int64_t>(handle.owner_pid));
    info.insert("device_id", static_cast<int64_t>(handle.device_id));
    return info;
}

bool unregister_kv_cache_block_gather_host_pool(int64_t opaque_handle)
{
    const int32_t device_id = get_current_host_mapping_device();
    std::lock_guard<std::mutex> guard(get_host_mapping_mutex());
    ensure_host_mapping_registry_process_locked();
    auto& handles = get_host_mapping_handles();
    const auto handle_it = handles.find(opaque_handle);
    if (handle_it == handles.end() || !handle_it->second.active) {
        return false;
    }

    auto& handle = handle_it->second;
    const auto mapping = find_host_mapping_by_id_locked(handle.mapping_id);
    TORCH_CHECK(mapping != nullptr,
                "kv_cache_block_gather: active host mapping handle has no mapping: ",
                opaque_handle);
    TORCH_CHECK(mapping->explicit_handle_count > 0,
                "kv_cache_block_gather: invalid explicit host mapping reference count");
    TORCH_CHECK(mapping->device_id == device_id,
                "kv_cache_block_gather: cannot release a host pool registered "
                "on NPU device ",
                mapping->device_id,
                " while current device is ",
                device_id);

    const bool remove_mapping = mapping->explicit_handle_count == 1;
    if (remove_mapping) {
        synchronize_before_host_mapping_release_locked();
    }
    if (remove_mapping && mapping->owned) {
        const aclError ret =
            aclrtHostUnregister(reinterpret_cast<void*>(mapping->host_base));
        TORCH_CHECK(ret == ACL_SUCCESS,
                    "kv_cache_block_gather: aclrtHostUnregister failed, error code: ",
                    ret,
                    ", host_base=",
                    reinterpret_cast<void*>(mapping->host_base),
                    ", register_size=",
                    mapping->size);
    }

    mapping->explicit_handle_count -= 1;
    auto& counters = get_host_mapping_counters();
    counters.explicit_unregister_call_count += 1;
    if (remove_mapping) {
        if (mapping->owned) {
            counters.unregister_call_count += 1;
            counters.register_bytes_current -= mapping->size;
        } else {
            // The allocator owns the borrowed physical registration.  The
            // device fence above is still required before erasing this handle,
            // because doing so drops tensor_owner.
        }
        auto& mappings = get_host_mappings();
        mappings.erase(
            std::remove_if(mappings.begin(),
                           mappings.end(),
                           [mapping_id = mapping->mapping_id](const auto& item) {
                               return item->mapping_id == mapping_id;
                           }),
            mappings.end());
    }
    handles.erase(handle_it);
    return true;
}

void kv_cache_block_gather(const torch::Tensor& src_block_ids,
                           const torch::Tensor& src_pages,
                           const torch::Tensor& dst_block_ids,
                           torch::Tensor& out)
{
    // Framework adapter, not the device algorithm.  Its responsibilities are:
    // validate the Torch contract, resolve a registered CPU source range, create ACL
    // tensor descriptors, query host tiling/workspace, and enqueue ACLNN on the
    // current NPU stream.  op_kernel/ performs the actual payload movement.
    TORCH_CHECK(src_block_ids.is_privateuseone(), "src_block_ids must be on NPU");
    TORCH_CHECK(dst_block_ids.is_privateuseone(), "dst_block_ids must be on NPU");
    TORCH_CHECK(out.is_privateuseone(), "out must be on NPU");
    TORCH_CHECK(src_block_ids.device() == out.device(),
                "src_block_ids and out must be on the same NPU device, got ",
                src_block_ids.device(),
                " and ",
                out.device());
    TORCH_CHECK(dst_block_ids.device() == out.device(),
                "dst_block_ids and out must be on the same NPU device, got ",
                dst_block_ids.device(),
                " and ",
                out.device());
    TORCH_CHECK(src_pages.device().is_cpu(), "src_pages must be a CPU tensor");
    TORCH_CHECK(src_block_ids.scalar_type() == at::ScalarType::Int, "src_block_ids must be int32");
    TORCH_CHECK(dst_block_ids.scalar_type() == at::ScalarType::Int, "dst_block_ids must be int32");
    TORCH_CHECK(src_block_ids.dim() == 1 && dst_block_ids.dim() == 1,
                "src_block_ids and dst_block_ids must be 1D");
    TORCH_CHECK(src_block_ids.size(0) == dst_block_ids.size(0),
                "src_block_ids and dst_block_ids length must match");
    if (src_block_ids.size(0) == 0) {
        return;
    }
    TORCH_CHECK(src_block_ids.is_contiguous(), "src_block_ids must be contiguous");
    TORCH_CHECK(dst_block_ids.is_contiguous(), "dst_block_ids must be contiguous");
    TORCH_CHECK(src_pages.is_contiguous(), "src_pages must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(src_pages.dim() >= 1 && out.dim() >= 1,
                "src_pages and out must have at least 1 dimension");
    TORCH_CHECK(src_pages.scalar_type() == out.scalar_type(),
                "src_pages and out dtype must match");
    TORCH_CHECK(src_pages.scalar_type() == at::ScalarType::Float ||
                    src_pages.scalar_type() == at::ScalarType::Half ||
                    src_pages.scalar_type() == at::ScalarType::BFloat16 ||
                    src_pages.scalar_type() == at::ScalarType::Char,
                "src_pages dtype must be float32, float16, bfloat16, or int8");
    TORCH_CHECK(out.size(0) > 0, "out dim0 must be positive");
    TORCH_CHECK(out.numel() % out.size(0) == 0,
                "out element count must be divisible by out dim0");
    constexpr int64_t data_copy_block_bytes = 32;
    const int64_t elements_per_block = out.numel() / out.size(0);
    const int64_t bytes_per_block = elements_per_block * out.element_size();
    TORCH_CHECK(bytes_per_block % data_copy_block_bytes == 0,
                "bytes per block must be a multiple of ",
                data_copy_block_bytes,
                " for block-aligned DataCopy, got ",
                bytes_per_block);

    const c10_npu::OptionalNPUGuard npu_guard(out.device());
    void* mapped_src_pages = get_mapped_host_device_ptr(
        src_pages.data_ptr(),
        static_cast<uint64_t>(src_pages.numel() * src_pages.element_size()));

    aclTensor* acl_src_block_ids = ConvertType(src_block_ids);
    aclTensor* acl_src_pages = create_acl_tensor_with_data(src_pages, mapped_src_pages);
    aclTensor* acl_dst_block_ids = ConvertType(dst_block_ids);
    aclTensor* acl_out = ConvertType(out);

    static const auto get_workspace_addr =
        get_host_gather_opapi_func_addr("aclnnKvCacheBlockGatherGetWorkspaceSize");
    static const auto op_api_addr = get_host_gather_opapi_func_addr("aclnnKvCacheBlockGather");
    TORCH_CHECK(get_workspace_addr != nullptr && op_api_addr != nullptr,
                "aclnnKvCacheBlockGather or aclnnKvCacheBlockGatherGetWorkspaceSize not found in op_api libraries");

    using GetWorkspaceFunc = int (*)(const aclTensor*,
                                     const aclTensor*,
                                     const aclTensor*,
                                     const aclTensor*,
                                     uint64_t*,
                                     aclOpExecutor**);
    auto get_workspace = reinterpret_cast<GetWorkspaceFunc>(get_workspace_addr);

    // ACLNN asks the op_host tiling implementation for this value.  The current
    // definition returns 16 MiB; keep that behavior explicit in measurements
    // even though the present device algorithm does not dereference workspace.
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const int workspace_status = get_workspace(acl_src_block_ids,
                                               acl_src_pages,
                                               acl_dst_block_ids,
                                               acl_out,
                                               &workspace_size,
                                               &executor);
    TORCH_CHECK(workspace_status == 0,
                "call aclnnKvCacheBlockGatherGetWorkspaceSize failed, detail:",
                aclGetRecentErrMsg());

    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        at::TensorOptions options =
            at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},
                                     options.dtype(at::kByte));
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    auto acl_call = [workspace_addr,
                     workspace_size,
                     executor,
                     acl_src_block_ids,
                     acl_src_pages,
                     acl_dst_block_ids,
                     acl_out]() -> int {
        using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
        auto op_api = reinterpret_cast<OpApiFunc>(op_api_addr);
        // Resolve the current stream inside the custom handler.  Timing harnesses
        // must synchronize completed device work, not assume a Python-side event
        // on an unrelated custom stream covers this submission.
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        const int ret = op_api(workspace_addr, workspace_size, executor, acl_stream);
        Release(acl_src_block_ids);
        Release(acl_src_pages);
        Release(acl_dst_block_ids);
        Release(acl_out);
        TORCH_CHECK(ret == 0,
                    "call aclnnKvCacheBlockGather failed, detail:",
                    aclGetRecentErrMsg());
        return ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnKvCacheBlockGather");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
}

} // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("kv_cache_block_gather(Tensor src_block_ids, Tensor src_pages, Tensor dst_block_ids, Tensor! out) -> ()");
    ops.impl("kv_cache_block_gather", torch::kPrivateUse1,
             &vllm_ascend::kv_cache_block_gather);
    ops.def("has_kv_cache_block_gather_runtime() -> bool");
    ops.impl("has_kv_cache_block_gather_runtime",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::has_kv_cache_block_gather_runtime);
    ops.def("load_kv_cache_block_gather_runtime(str path) -> bool");
    ops.impl("load_kv_cache_block_gather_runtime",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::load_kv_cache_block_gather_runtime);
    ops.def("is_kv_cache_block_gather_host_mapping_cached(Tensor src_pages) -> bool");
    ops.impl("is_kv_cache_block_gather_host_mapping_cached",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::is_kv_cache_block_gather_host_mapping_cached);
    ops.def("register_kv_cache_block_gather_host_pool(Tensor src_pages) -> int");
    ops.impl("register_kv_cache_block_gather_host_pool",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::register_kv_cache_block_gather_host_pool);
    ops.def("inspect_kv_cache_block_gather_host_pool(int handle) -> Dict(str, int)");
    ops.impl("inspect_kv_cache_block_gather_host_pool",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::inspect_kv_cache_block_gather_host_pool);
    ops.def("unregister_kv_cache_block_gather_host_pool(int handle) -> bool");
    ops.impl("unregister_kv_cache_block_gather_host_pool",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::unregister_kv_cache_block_gather_host_pool);
    ops.def("get_kv_cache_block_gather_host_mapping_stats() -> Dict(str, int)");
    ops.impl("get_kv_cache_block_gather_host_mapping_stats",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_kv_cache_block_gather_host_mapping_stats);
}
