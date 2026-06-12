#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"
#include <Metal/Metal.hpp>
#include <unistd.h>
#include <unordered_map>
#include <mutex>

namespace py = pybind11;
using namespace mlx::core;

struct CachedBuffer {
    MTL::Buffer* buffer;
    int64_t offset;
};

// Global cache for static pools (theta, m, v, indices) to eliminate newBuffer CPU-side allocation overhead
std::unordered_map<unsigned long, CachedBuffer> buffer_cache;
std::mutex cache_mutex;

// Asynchronous delayed release queue to resolve GPU Invalid Resource race conditions
std::vector<MTL::Buffer*> pending_releases;
std::mutex release_mutex;

void clear_buffer_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    std::lock_guard<std::mutex> r_lock(release_mutex);
    for (auto& pair : buffer_cache) {
        if (pair.second.buffer) {
            pending_releases.push_back(pair.second.buffer);
        }
    }
    buffer_cache.clear();
}

/**
 * Helper to safely wrap potentially unaligned raw pointers from MLX/NumPy
 * into page-aligned MTL::Buffer handles using virtual memory page alignment.
 * If cacheable is true, it reuses the pre-created buffer from global cache.
 */
CachedBuffer get_or_create_safe_buffer(MTL::Device* mtl_dev, unsigned long addr, size_t size, bool cacheable) {
    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = buffer_cache.find(addr);
        if (it != buffer_cache.end()) {
            return it->second;
        }
    }

    size_t page_size = getpagesize();
    uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
    uintptr_t aligned_addr = ptr & ~(page_size - 1);
    int64_t offset = ptr - aligned_addr;
    size_t aligned_length = size + offset;
    
    MTL::Buffer* buf = mtl_dev->newBuffer(reinterpret_cast<void*>(aligned_addr), aligned_length, MTL::ResourceStorageModeShared);
    if (!buf) {
        throw std::runtime_error("[METAL ERROR] Failed to create MTL::Buffer for address: " + std::to_string(addr) + " with size: " + std::to_string(size));
    }
    
    CachedBuffer cb = {buf, offset};
    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        buffer_cache[addr] = cb;
    }
    return cb;
}

/**
 * Highly optimized C++ entrypoint utilizing UMA Shared DRAM address wrapping
 * to bypass pybind11 ABI symbol matching issues and perform TRUE in-place GPU updates.
 */
void launch_fused_adam_step(
    unsigned long theta_addr,       // Entire physical pool [max_gaussians, dim]
    unsigned long g_addr,           // Active gradients [num_active, dim]
    unsigned long m_addr,           // Entire physical m pool [max_gaussians, dim]
    unsigned long v_addr,           // Entire physical v pool [max_gaussians, dim]
    unsigned long indices_addr,     // Active indices indirection table [num_active]
    size_t max_gaussians,
    size_t num_active,
    int dim,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    float a_min,
    float a_max
) {
    // 1. Get the Metal stream and device context from MLX
    auto& metal_dev = mlx::core::metal::device(default_device());
    MTL::Device* mtl_dev = metal_dev.mtl_device();
    
    // 2. Load the MSL kernel source code
    const std::string msl_source = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void fused_adam_step(
            device float* theta            [[buffer(0)]], // Physical pool
            device const float* g          [[buffer(1)]], // Active gradients
            device float* m                [[buffer(2)]], // Physical m pool
            device float* v                [[buffer(3)]], // Physical v pool
            device const int* indices      [[buffer(4)]], // Indirection table
            constant float& lr             [[buffer(5)]],
            constant float& beta1          [[buffer(6)]],
            constant float& beta2          [[buffer(7)]],
            constant float& eps            [[buffer(8)]],
            constant float& weight_decay   [[buffer(9)]],
            constant int& step             [[buffer(10)]],
            constant int& dim              [[buffer(11)]],
            constant float& a_min          [[buffer(12)]],
            constant float& a_max          [[buffer(13)]],
            uint id                        [[thread_position_in_grid]]
        ) {
            int active_idx = id / dim;
            int d = id % dim;
            int physical_id = indices[active_idx] * dim + d;

            float grad = g[id];
            if (isnan(grad)) {
                grad = 0.0f;
            }
            
            if (weight_decay != 0.0f) {
                grad += weight_decay * theta[physical_id];
            }
            float m_new = beta1 * m[physical_id] + (1.0f - beta1) * grad;
            float v_new = beta2 * v[physical_id] + (1.0f - beta2) * (grad * grad);
            
            if (isnan(m_new) || isinf(m_new)) m_new = 0.0f;
            if (isnan(v_new) || isinf(v_new)) v_new = 0.0f;
            
            m[physical_id] = m_new;
            v[physical_id] = v_new;
            
            float bias_correction1 = 1.0f - pow(beta1, (float)step);
            float bias_correction2 = 1.0f - pow(beta2, (float)step);
            
            float step_size = lr / bias_correction1;
            float denom = (sqrt(v_new) / sqrt(bias_correction2)) + eps;
            
            float delta = step_size * (m_new / denom);
            if (isnan(delta) || isinf(delta)) delta = 0.0f;
            
            float new_val = theta[physical_id] - delta;
            
            theta[physical_id] = clamp(new_val, a_min, a_max);
        }
    )";

    // 3. Compile/retrieve the metal pipeline state using dynamic lambda compilation
    auto lib = metal_dev.get_library("mlx_gsplat_ext", [msl_source]() {
        return msl_source;
    });
    auto kernel = metal_dev.get_kernel("fused_adam_step", lib);

    // 4. Wrap UMA raw pointers safely with page alignment offsets (Using high-speed caching)
    CachedBuffer theta_cb = get_or_create_safe_buffer(mtl_dev, theta_addr, max_gaussians * dim * sizeof(float), true);
    CachedBuffer g_cb = get_or_create_safe_buffer(mtl_dev, g_addr, num_active * dim * sizeof(float), false);
    CachedBuffer m_cb = get_or_create_safe_buffer(mtl_dev, m_addr, max_gaussians * dim * sizeof(float), true);
    CachedBuffer v_cb = get_or_create_safe_buffer(mtl_dev, v_addr, max_gaussians * dim * sizeof(float), true);
    CachedBuffer indices_cb = get_or_create_safe_buffer(mtl_dev, indices_addr, num_active * sizeof(int), true);

    MTL::Buffer* theta_buf = theta_cb.buffer;
    MTL::Buffer* g_buf = g_cb.buffer;
    MTL::Buffer* m_buf = m_cb.buffer;
    MTL::Buffer* v_buf = v_cb.buffer;
    MTL::Buffer* indices_buf = indices_cb.buffer;

    int64_t theta_offset = theta_cb.offset;
    int64_t g_offset = g_cb.offset;
    int64_t m_offset = m_cb.offset;
    int64_t v_offset = v_cb.offset;
    int64_t indices_offset = indices_cb.offset;

    // 5. Get command encoder and dispatch via UMA-Native MLX CommandEncoder
    auto& encoder = mlx::core::metal::get_command_encoder(default_stream(default_device()));
    
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_buffer(theta_buf, 0, theta_offset);
    encoder.set_buffer(g_buf, 1, g_offset);
    encoder.set_buffer(m_buf, 2, m_offset);
    encoder.set_buffer(v_buf, 3, v_offset);
    encoder.set_buffer(indices_buf, 4, indices_offset);
    encoder.set_bytes(lr, 5);
    encoder.set_bytes(beta1, 6);
    encoder.set_bytes(beta2, 7);
    encoder.set_bytes(eps, 8);
    encoder.set_bytes(weight_decay, 9);
    encoder.set_bytes(step, 10);
    encoder.set_bytes(dim, 11);
    encoder.set_bytes(a_min, 12);
    encoder.set_bytes(a_max, 13);

    // Dispatch grid matching the parameter element count exactly
    size_t num_elements = num_active * dim;
    MTL::Size grid_size(num_elements, 1, 1);
    MTL::Size group_size(std::min(num_elements, size_t(256)), 1, 1);
    encoder.dispatch_threads(grid_size, group_size);

    // Register an asynchronous completion handler to release the temporary handles 
    // when the GPU has finished executing this command buffer, avoiding CPU blocking!
    // Note: Cached buffers are retained and cleared only during Densification to bypass VM overhead.
    // Delayed releases are executed fully asynchronously on GPU completion to avoid Invalid Resource race conditions.
    auto cmd_buf = encoder.get_command_buffer();
    
    std::vector<MTL::Buffer*> to_release;
    to_release.push_back(g_buf); // Always release the temporary gradient buffer
    
    {
        std::lock_guard<std::mutex> r_lock(release_mutex);
        if (!pending_releases.empty()) {
            to_release.insert(to_release.end(), pending_releases.begin(), pending_releases.end());
            pending_releases.clear();
        }
    }
    
    if (cmd_buf) {
        cmd_buf->addCompletedHandler(
            [to_release](MTL::CommandBuffer* cb) {
                for (auto* buf : to_release) {
                    if (buf) buf->release();
                }
            }
        );
    } else {
        // Fallback safety: Release synchronously if command buffer is not present
        for (auto* buf : to_release) {
            if (buf) buf->release();
        }
    }
}

PYBIND11_MODULE(mlx_gsplat_ext, m) {
    m.doc() = "MLX custom extension for highly optimized UMA 3DGS kernels";
    m.def("launch_fused_adam_step", &launch_fused_adam_step, 
          "Launches the fused Metal Adam optimization step kernel in-place on Unified DRAM.",
          py::arg("theta_addr"), py::arg("g_addr"), py::arg("m_addr"), py::arg("v_addr"), 
          py::arg("indices_addr"), py::arg("max_gaussians"), py::arg("num_active"), py::arg("dim"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), 
          py::arg("weight_decay"), py::arg("step"), py::arg("a_min"), py::arg("a_max"));
    m.def("clear_buffer_cache", &clear_buffer_cache,
          "Clears the cached Metal buffers to prevent memory leaks and handle pool resizing.");
}
