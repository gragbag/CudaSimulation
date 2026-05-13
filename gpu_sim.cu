#include "gpu_sim.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>

// srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=16 --cpus-per-task=1 --mem=20g --pty bash
// make
// ./particle_sim gpu_tiled 100000 500 10 combined

static void check_cuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(result)
        );
    }
}

class ScopedCudaEvents {
public:
    ScopedCudaEvents() {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate start failed");
        check_cuda(cudaEventCreate(&stop_), "cudaEventCreate stop failed");
    }

    ~ScopedCudaEvents() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    template <typename Func>
    float measure(Func&& func) {
        check_cuda(cudaEventRecord(start_), "cudaEventRecord start failed");
        func();
        check_cuda(cudaEventRecord(stop_), "cudaEventRecord stop failed");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize failed");

        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_, stop_),
                   "cudaEventElapsedTime failed");
        return elapsed_ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

static void print_timings(
    const char* label,
    const GpuTimings& timings,
    int blocks,
    int threads_per_block,
    size_t shared_mem_bytes = 0
) {
    const float total_ms =
        timings.malloc_ms + timings.h2d_ms + timings.kernel_ms + timings.d2h_ms;

    std::cout << "\n" << label << "\n";
    std::cout << "    cudaMalloc:                      "
              << timings.malloc_ms / 1000.0f << " s\n";
    std::cout << "    cudaMemcpy H2D:                  "
              << timings.h2d_ms / 1000.0f << " s\n";
    std::cout << "    kernel<<<" << blocks << ", " << threads_per_block;

    if (shared_mem_bytes > 0) {
        std::cout << ", " << shared_mem_bytes;
    }

    std::cout << ">>>:              " << timings.kernel_ms / 1000.0f << " s\n";
    std::cout << "    cudaMemcpy D2H:                  "
              << timings.d2h_ms / 1000.0f << " s\n";
    std::cout << "    Total GPU time:                  "
              << total_ms / 1000.0f << " s\n";
}

// 0 = gravity, 1 = electrostatic, 2 = combined
__device__ float compute_force_scale(
    float mi,
    float mj,
    float qi,
    float qj,
    float inv_dist3,
    int force_mode
) {
    float scale = 0.0f;

    if (force_mode == 0 || force_mode == 2) {
        scale += G * mi * mj * inv_dist3;
    }

    if (force_mode == 1 || force_mode == 2) {
        scale += -K_E * qi * qj * inv_dist3;
    }

    return scale;
}

__global__ void simulate_step_kernel_sqrt(
    const Particle* old_particles,
    Particle* new_particles,
    int n,
    int force_mode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    float x = old_particles[i].x;
    float y = old_particles[i].y;
    float z = old_particles[i].z;
    float vx = old_particles[i].vx;
    float vy = old_particles[i].vy;
    float vz = old_particles[i].vz;
    float mass = old_particles[i].mass;
    float charge = old_particles[i].charge;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        float dx = old_particles[j].x - x;
        float dy = old_particles[j].y - y;
        float dz = old_particles[j].z - z;

        float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float dist = sqrtf(dist_sqr);
        float inv_dist3 = 1.0f / (dist_sqr * dist);

        float scale = compute_force_scale(
            mass,
            old_particles[j].mass,
            charge,
            old_particles[j].charge,
            inv_dist3,
            force_mode
        );

        fx += scale * dx;
        fy += scale * dy;
        fz += scale * dz;
    }

    float ax = fx / mass;
    float ay = fy / mass;
    float az = fz / mass;

    Particle updated;
    updated.vx = vx + ax * DT;
    updated.vy = vy + ay * DT;
    updated.vz = vz + az * DT;

    updated.x = x + updated.vx * DT;
    updated.y = y + updated.vy * DT;
    updated.z = z + updated.vz * DT;

    updated.mass = mass;
    updated.charge = charge;

    new_particles[i] = updated;
}

__global__ void simulate_step_kernel(
    const Particle* old_particles,
    Particle* new_particles,
    int n,
    int force_mode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    float x = old_particles[i].x;
    float y = old_particles[i].y;
    float z = old_particles[i].z;
    float vx = old_particles[i].vx;
    float vy = old_particles[i].vy;
    float vz = old_particles[i].vz;
    float mass = old_particles[i].mass;
    float charge = old_particles[i].charge;

    for (int j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }

        float dx = old_particles[j].x - x;
        float dy = old_particles[j].y - y;
        float dz = old_particles[j].z - z;

        float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float inv_dist = rsqrtf(dist_sqr);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;

        float scale = compute_force_scale(
            mass,
            old_particles[j].mass,
            charge,
            old_particles[j].charge,
            inv_dist3,
            force_mode
        );

        fx += scale * dx;
        fy += scale * dy;
        fz += scale * dz;
    }

    float ax = fx / mass;
    float ay = fy / mass;
    float az = fz / mass;

    Particle updated;
    updated.vx = vx + ax * DT;
    updated.vy = vy + ay * DT;
    updated.vz = vz + az * DT;

    updated.x = x + updated.vx * DT;
    updated.y = y + updated.vy * DT;
    updated.z = z + updated.vz * DT;

    updated.mass = mass;
    updated.charge = charge;

    new_particles[i] = updated;
}

__global__ void simulate_step_tiled_kernel(
    const Particle* old_particles,
    Particle* new_particles,
    int n,
    int force_mode
) {
    extern __shared__ Particle tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float mass = 1.0f;
    float charge = 0.0f;

    bool active = (i < n);

    if (active) {
        x = old_particles[i].x;
        y = old_particles[i].y;
        z = old_particles[i].z;
        vx = old_particles[i].vx;
        vy = old_particles[i].vy;
        vz = old_particles[i].vz;
        mass = old_particles[i].mass;
        charge = old_particles[i].charge;
    }

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int j = tile_start + tid;

        if (j < n) {
            tile[tid] = old_particles[j];
        }

        __syncthreads();

        if (active) {
            int tile_size = min(blockDim.x, n - tile_start);

            for (int offset = 0; offset < tile_size; ++offset) {
                int global_j = tile_start + offset;
                if (global_j == i) continue;

                Particle other = tile[offset];

                float dx = other.x - x;
                float dy = other.y - y;
                float dz = other.z - z;

                float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float inv_dist = rsqrtf(dist_sqr);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;

                float scale = compute_force_scale(
                    mass,
                    other.mass,
                    charge,
                    other.charge,
                    inv_dist3,
                    force_mode
                );

                fx += scale * dx;
                fy += scale * dy;
                fz += scale * dz;
            }
        }

        __syncthreads();
    }

    if (active) {
        float ax = fx / mass;
        float ay = fy / mass;
        float az = fz / mass;

        Particle updated;
        updated.vx = vx + ax * DT;
        updated.vy = vy + ay * DT;
        updated.vz = vz + az * DT;

        updated.x = x + updated.vx * DT;
        updated.y = y + updated.vy * DT;
        updated.z = z + updated.vz * DT;
        updated.mass = mass;
        updated.charge = charge;

        new_particles[i] = updated;
    }
}

__global__ void simulate_step_tiled_kernel_sqrt(
    const Particle* old_particles,
    Particle* new_particles,
    int n,
    int force_mode
) {
    extern __shared__ Particle tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float mass = 1.0f;
    float charge = 0.0f;

    bool active = (i < n);

    if (active) {
        x = old_particles[i].x;
        y = old_particles[i].y;
        z = old_particles[i].z;
        vx = old_particles[i].vx;
        vy = old_particles[i].vy;
        vz = old_particles[i].vz;
        mass = old_particles[i].mass;
        charge = old_particles[i].charge;
    }

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int j = tile_start + tid;

        if (j < n) {
            tile[tid] = old_particles[j];
        }

        __syncthreads();

        if (active) {
            int tile_size = min(blockDim.x, n - tile_start);

            for (int offset = 0; offset < tile_size; ++offset) {
                int global_j = tile_start + offset;
                if (global_j == i) continue;

                Particle other = tile[offset];

                float dx = other.x - x;
                float dy = other.y - y;
                float dz = other.z - z;

                float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float dist = sqrtf(dist_sqr);
                float inv_dist3 = 1.0f / (dist_sqr * dist);

                float scale = compute_force_scale(
                    mass,
                    other.mass,
                    charge,
                    other.charge,
                    inv_dist3,
                    force_mode
                );

                fx += scale * dx;
                fy += scale * dy;
                fz += scale * dz;
            }
        }

        __syncthreads();
    }

    if (active) {
        float ax = fx / mass;
        float ay = fy / mass;
        float az = fz / mass;

        Particle updated;
        updated.vx = vx + ax * DT;
        updated.vy = vy + ay * DT;
        updated.vz = vz + az * DT;

        updated.x = x + updated.vx * DT;
        updated.y = y + updated.vy * DT;
        updated.z = z + updated.vz * DT;
        updated.mass = mass;
        updated.charge = charge;

        new_particles[i] = updated;
    }
}

__global__ void simulate_step_soa_kernel(
    const float* old_x,
    const float* old_y,
    const float* old_z,
    const float* old_vx,
    const float* old_vy,
    const float* old_vz,
    const float* mass,
    const float* charge,
    float* new_x,
    float* new_y,
    float* new_z,
    float* new_vx,
    float* new_vy,
    float* new_vz,
    int n,
    int force_mode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float xi = old_x[i];
    float yi = old_y[i];
    float zi = old_z[i];

    float vxi = old_vx[i];
    float vyi = old_vy[i];
    float vzi = old_vz[i];

    float mi = mass[i];
    float qi = charge[i];

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }

        float dx = old_x[j] - xi;
        float dy = old_y[j] - yi;
        float dz = old_z[j] - zi;

        float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;

        float inv_dist = rsqrtf(dist_sqr);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;

        float scale = compute_force_scale(
            mi,
            mass[j],
            qi,
            charge[j],
            inv_dist3,
            force_mode
        );

        fx += scale * dx;
        fy += scale * dy;
        fz += scale * dz;
    }

    float ax = fx / mi;
    float ay = fy / mi;
    float az = fz / mi;

    float next_vx = vxi + ax * DT;
    float next_vy = vyi + ay * DT;
    float next_vz = vzi + az * DT;

    new_vx[i] = next_vx;
    new_vy[i] = next_vy;
    new_vz[i] = next_vz;

    new_x[i] = xi + next_vx * DT;
    new_y[i] = yi + next_vy * DT;
    new_z[i] = zi + next_vz * DT;
}

__global__ void simulate_step_soa_tiled_kernel(
    const float* old_x,
    const float* old_y,
    const float* old_z,
    const float* old_vx,
    const float* old_vy,
    const float* old_vz,
    const float* mass,
    const float* charge,
    float* new_x,
    float* new_y,
    float* new_z,
    float* new_vx,
    float* new_vy,
    float* new_vz,
    int n,
    int force_mode
) {
    extern __shared__ float shared[];

    float* tile_x = shared;
    float* tile_y = tile_x + blockDim.x;
    float* tile_z = tile_y + blockDim.x;
    float* tile_m = tile_z + blockDim.x;
    float* tile_q = tile_m + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    bool active = (i < n);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    float mi = 1.0f;
    float qi = 0.0f;

    if (active) {
        xi = old_x[i];
        yi = old_y[i];
        zi = old_z[i];

        vxi = old_vx[i];
        vyi = old_vy[i];
        vzi = old_vz[i];

        mi = mass[i];
        qi = charge[i];
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int j = tile_start + tid;

        if (j < n) {
            tile_x[tid] = old_x[j];
            tile_y[tid] = old_y[j];
            tile_z[tid] = old_z[j];
            tile_m[tid] = mass[j];
            tile_q[tid] = charge[j];
        }

        __syncthreads();

        if (active) {
            int tile_size = min(blockDim.x, n - tile_start);

            for (int offset = 0; offset < tile_size; ++offset) {
                int global_j = tile_start + offset;

                if (global_j == i) {
                    continue;
                }

                float dx = tile_x[offset] - xi;
                float dy = tile_y[offset] - yi;
                float dz = tile_z[offset] - zi;

                float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;

                float inv_dist = rsqrtf(dist_sqr);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;

                float scale = compute_force_scale(
                    mi,
                    tile_m[offset],
                    qi,
                    tile_q[offset],
                    inv_dist3,
                    force_mode
                );

                fx += scale * dx;
                fy += scale * dy;
                fz += scale * dz;
            }
        }

        __syncthreads();
    }

    if (active) {
        float ax = fx / mi;
        float ay = fy / mi;
        float az = fz / mi;

        float next_vx = vxi + ax * DT;
        float next_vy = vyi + ay * DT;
        float next_vz = vzi + az * DT;

        new_vx[i] = next_vx;
        new_vy[i] = next_vy;
        new_vz[i] = next_vz;

        new_x[i] = xi + next_vx * DT;
        new_y[i] = yi + next_vy * DT;
        new_z[i] = zi + next_vz * DT;
    }
}

void simulate_gpu_sqrt(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    Particle* d_particles_a = nullptr;
    Particle* d_particles_b = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(Particle);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_particles_a, bytes), "cudaMalloc d_particles_a failed");
        check_cuda(cudaMalloc(&d_particles_b, bytes), "cudaMalloc d_particles_b failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_particles_a, particles.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device failed");
    });

    Particle* d_old = d_particles_a;
    Particle* d_new = d_particles_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_kernel_sqrt<<<blocks, threads_per_block>>>(d_old, d_new, n, force_mode);

            check_cuda(cudaGetLastError(), "Kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

            std::swap(d_old, d_new);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(particles.data(), d_old, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host failed");
    });

    if (print_timing) {
        print_timings("Particle sim on GPU (AoS, sqrt)", timings, blocks, threads_per_block);
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_particles_a);
    cudaFree(d_particles_b);
}

void simulate_gpu(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    Particle* d_particles_a = nullptr;
    Particle* d_particles_b = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(Particle);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_particles_a, bytes), "cudaMalloc d_particles_a failed");
        check_cuda(cudaMalloc(&d_particles_b, bytes), "cudaMalloc d_particles_b failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_particles_a, particles.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device failed");
    });

    Particle* d_old = d_particles_a;
    Particle* d_new = d_particles_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_kernel<<<blocks, threads_per_block>>>(d_old, d_new, n, force_mode);

            check_cuda(cudaGetLastError(), "Kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

            std::swap(d_old, d_new);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(particles.data(), d_old, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host failed");
    });

    if (print_timing) {
        print_timings("Particle sim on GPU (AoS, rsqrt)", timings, blocks, threads_per_block);
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_particles_a);
    cudaFree(d_particles_b);
}

void simulate_gpu_tiled(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    Particle* d_particles_a = nullptr;
    Particle* d_particles_b = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(Particle);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_bytes = threads_per_block * sizeof(Particle);

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_particles_a, bytes), "cudaMalloc d_particles_a failed");
        check_cuda(cudaMalloc(&d_particles_b, bytes), "cudaMalloc d_particles_b failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_particles_a, particles.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device failed");
    });

    Particle* d_old = d_particles_a;
    Particle* d_new = d_particles_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
                d_old, d_new, n, force_mode
            );

            check_cuda(cudaGetLastError(), "Tiled kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "Tiled kernel execution failed");

            std::swap(d_old, d_new);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(particles.data(), d_old, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host failed");
    });

    if (print_timing) {
        print_timings(
            "Particle sim on GPU (AoS, tiled)",
            timings,
            blocks,
            threads_per_block,
            shared_mem_bytes
        );
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_particles_a);
    cudaFree(d_particles_b);
}

void simulate_gpu_tiled_sqrt(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    Particle* d_particles_a = nullptr;
    Particle* d_particles_b = nullptr;

    size_t bytes = static_cast<size_t>(n) * sizeof(Particle);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_bytes = threads_per_block * sizeof(Particle);

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_particles_a, bytes), "cudaMalloc d_particles_a failed");
        check_cuda(cudaMalloc(&d_particles_b, bytes), "cudaMalloc d_particles_b failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_particles_a, particles.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device failed");
    });

    Particle* d_old = d_particles_a;
    Particle* d_new = d_particles_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_tiled_kernel_sqrt<<<blocks, threads_per_block, shared_mem_bytes>>>(
                d_old, d_new, n, force_mode
            );

            check_cuda(cudaGetLastError(), "Tiled sqrt kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "Tiled sqrt kernel execution failed");

            std::swap(d_old, d_new);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(particles.data(), d_old, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host failed");
    });

    if (print_timing) {
        print_timings(
            "Particle sim on GPU (AoS, tiled, sqrt)",
            timings,
            blocks,
            threads_per_block,
            shared_mem_bytes
        );
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_particles_a);
    cudaFree(d_particles_b);
}

void simulate_gpu_soa(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    std::vector<float> h_x(n), h_y(n), h_z(n);
    std::vector<float> h_vx(n), h_vy(n), h_vz(n);
    std::vector<float> h_mass(n), h_charge(n);

    for (int i = 0; i < n; ++i) {
        h_x[i] = particles[i].x;
        h_y[i] = particles[i].y;
        h_z[i] = particles[i].z;

        h_vx[i] = particles[i].vx;
        h_vy[i] = particles[i].vy;
        h_vz[i] = particles[i].vz;

        h_mass[i] = particles[i].mass;
        h_charge[i] = particles[i].charge;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_x_a = nullptr, *d_y_a = nullptr, *d_z_a = nullptr;
    float *d_vx_a = nullptr, *d_vy_a = nullptr, *d_vz_a = nullptr;

    float *d_x_b = nullptr, *d_y_b = nullptr, *d_z_b = nullptr;
    float *d_vx_b = nullptr, *d_vy_b = nullptr, *d_vz_b = nullptr;

    float *d_mass = nullptr;
    float *d_charge = nullptr;

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_x_a, bytes), "cudaMalloc d_x_a failed");
        check_cuda(cudaMalloc(&d_y_a, bytes), "cudaMalloc d_y_a failed");
        check_cuda(cudaMalloc(&d_z_a, bytes), "cudaMalloc d_z_a failed");
        check_cuda(cudaMalloc(&d_vx_a, bytes), "cudaMalloc d_vx_a failed");
        check_cuda(cudaMalloc(&d_vy_a, bytes), "cudaMalloc d_vy_a failed");
        check_cuda(cudaMalloc(&d_vz_a, bytes), "cudaMalloc d_vz_a failed");

        check_cuda(cudaMalloc(&d_x_b, bytes), "cudaMalloc d_x_b failed");
        check_cuda(cudaMalloc(&d_y_b, bytes), "cudaMalloc d_y_b failed");
        check_cuda(cudaMalloc(&d_z_b, bytes), "cudaMalloc d_z_b failed");
        check_cuda(cudaMalloc(&d_vx_b, bytes), "cudaMalloc d_vx_b failed");
        check_cuda(cudaMalloc(&d_vy_b, bytes), "cudaMalloc d_vy_b failed");
        check_cuda(cudaMalloc(&d_vz_b, bytes), "cudaMalloc d_vz_b failed");

        check_cuda(cudaMalloc(&d_mass, bytes), "cudaMalloc d_mass failed");
        check_cuda(cudaMalloc(&d_charge, bytes), "cudaMalloc d_charge failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_x_a, h_x.data(), bytes, cudaMemcpyHostToDevice), "copy x failed");
        check_cuda(cudaMemcpy(d_y_a, h_y.data(), bytes, cudaMemcpyHostToDevice), "copy y failed");
        check_cuda(cudaMemcpy(d_z_a, h_z.data(), bytes, cudaMemcpyHostToDevice), "copy z failed");

        check_cuda(cudaMemcpy(d_vx_a, h_vx.data(), bytes, cudaMemcpyHostToDevice), "copy vx failed");
        check_cuda(cudaMemcpy(d_vy_a, h_vy.data(), bytes, cudaMemcpyHostToDevice), "copy vy failed");
        check_cuda(cudaMemcpy(d_vz_a, h_vz.data(), bytes, cudaMemcpyHostToDevice), "copy vz failed");

        check_cuda(cudaMemcpy(d_mass, h_mass.data(), bytes, cudaMemcpyHostToDevice), "copy mass failed");
        check_cuda(cudaMemcpy(d_charge, h_charge.data(), bytes, cudaMemcpyHostToDevice), "copy charge failed");
    });

    float *old_x = d_x_a, *old_y = d_y_a, *old_z = d_z_a;
    float *old_vx = d_vx_a, *old_vy = d_vy_a, *old_vz = d_vz_a;

    float *new_x = d_x_b, *new_y = d_y_b, *new_z = d_z_b;
    float *new_vx = d_vx_b, *new_vy = d_vy_b, *new_vz = d_vz_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_soa_kernel<<<blocks, threads_per_block>>>(
                old_x, old_y, old_z,
                old_vx, old_vy, old_vz,
                d_mass,
                d_charge,
                new_x, new_y, new_z,
                new_vx, new_vy, new_vz,
                n,
                force_mode
            );

            check_cuda(cudaGetLastError(), "SoA kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "SoA kernel execution failed");

            std::swap(old_x, new_x);
            std::swap(old_y, new_y);
            std::swap(old_z, new_z);

            std::swap(old_vx, new_vx);
            std::swap(old_vy, new_vy);
            std::swap(old_vz, new_vz);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(h_x.data(), old_x, bytes, cudaMemcpyDeviceToHost), "copy x back failed");
        check_cuda(cudaMemcpy(h_y.data(), old_y, bytes, cudaMemcpyDeviceToHost), "copy y back failed");
        check_cuda(cudaMemcpy(h_z.data(), old_z, bytes, cudaMemcpyDeviceToHost), "copy z back failed");

        check_cuda(cudaMemcpy(h_vx.data(), old_vx, bytes, cudaMemcpyDeviceToHost), "copy vx back failed");
        check_cuda(cudaMemcpy(h_vy.data(), old_vy, bytes, cudaMemcpyDeviceToHost), "copy vy back failed");
        check_cuda(cudaMemcpy(h_vz.data(), old_vz, bytes, cudaMemcpyDeviceToHost), "copy vz back failed");
    });

    for (int i = 0; i < n; ++i) {
        particles[i].x = h_x[i];
        particles[i].y = h_y[i];
        particles[i].z = h_z[i];

        particles[i].vx = h_vx[i];
        particles[i].vy = h_vy[i];
        particles[i].vz = h_vz[i];

        particles[i].mass = h_mass[i];
        particles[i].charge = h_charge[i];
    }

    if (print_timing) {
        print_timings("Particle sim on GPU (SoA)", timings, blocks, threads_per_block);
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_x_a);  cudaFree(d_y_a);  cudaFree(d_z_a);
    cudaFree(d_vx_a); cudaFree(d_vy_a); cudaFree(d_vz_a);

    cudaFree(d_x_b);  cudaFree(d_y_b);  cudaFree(d_z_b);
    cudaFree(d_vx_b); cudaFree(d_vy_b); cudaFree(d_vz_b);

    cudaFree(d_mass);
    cudaFree(d_charge);
}

void simulate_gpu_soa_tiled(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing,
    GpuTimings* timings_out
) {
    int n = static_cast<int>(particles.size());
    if (n == 0 || num_steps <= 0) return;

    std::vector<float> h_x(n), h_y(n), h_z(n);
    std::vector<float> h_vx(n), h_vy(n), h_vz(n);
    std::vector<float> h_mass(n), h_charge(n);

    for (int i = 0; i < n; ++i) {
        h_x[i] = particles[i].x;
        h_y[i] = particles[i].y;
        h_z[i] = particles[i].z;

        h_vx[i] = particles[i].vx;
        h_vy[i] = particles[i].vy;
        h_vz[i] = particles[i].vz;

        h_mass[i] = particles[i].mass;
        h_charge[i] = particles[i].charge;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_x_a = nullptr, *d_y_a = nullptr, *d_z_a = nullptr;
    float *d_vx_a = nullptr, *d_vy_a = nullptr, *d_vz_a = nullptr;

    float *d_x_b = nullptr, *d_y_b = nullptr, *d_z_b = nullptr;
    float *d_vx_b = nullptr, *d_vy_b = nullptr, *d_vz_b = nullptr;

    float *d_mass = nullptr;
    float *d_charge = nullptr;

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_bytes = 5 * threads_per_block * sizeof(float);

    ScopedCudaEvents timer;
    GpuTimings timings;

    timings.malloc_ms = timer.measure([&] {
        check_cuda(cudaMalloc(&d_x_a, bytes), "cudaMalloc d_x_a failed");
        check_cuda(cudaMalloc(&d_y_a, bytes), "cudaMalloc d_y_a failed");
        check_cuda(cudaMalloc(&d_z_a, bytes), "cudaMalloc d_z_a failed");
        check_cuda(cudaMalloc(&d_vx_a, bytes), "cudaMalloc d_vx_a failed");
        check_cuda(cudaMalloc(&d_vy_a, bytes), "cudaMalloc d_vy_a failed");
        check_cuda(cudaMalloc(&d_vz_a, bytes), "cudaMalloc d_vz_a failed");

        check_cuda(cudaMalloc(&d_x_b, bytes), "cudaMalloc d_x_b failed");
        check_cuda(cudaMalloc(&d_y_b, bytes), "cudaMalloc d_y_b failed");
        check_cuda(cudaMalloc(&d_z_b, bytes), "cudaMalloc d_z_b failed");
        check_cuda(cudaMalloc(&d_vx_b, bytes), "cudaMalloc d_vx_b failed");
        check_cuda(cudaMalloc(&d_vy_b, bytes), "cudaMalloc d_vy_b failed");
        check_cuda(cudaMalloc(&d_vz_b, bytes), "cudaMalloc d_vz_b failed");

        check_cuda(cudaMalloc(&d_mass, bytes), "cudaMalloc d_mass failed");
        check_cuda(cudaMalloc(&d_charge, bytes), "cudaMalloc d_charge failed");
    });

    timings.h2d_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(d_x_a, h_x.data(), bytes, cudaMemcpyHostToDevice), "copy x failed");
        check_cuda(cudaMemcpy(d_y_a, h_y.data(), bytes, cudaMemcpyHostToDevice), "copy y failed");
        check_cuda(cudaMemcpy(d_z_a, h_z.data(), bytes, cudaMemcpyHostToDevice), "copy z failed");

        check_cuda(cudaMemcpy(d_vx_a, h_vx.data(), bytes, cudaMemcpyHostToDevice), "copy vx failed");
        check_cuda(cudaMemcpy(d_vy_a, h_vy.data(), bytes, cudaMemcpyHostToDevice), "copy vy failed");
        check_cuda(cudaMemcpy(d_vz_a, h_vz.data(), bytes, cudaMemcpyHostToDevice), "copy vz failed");

        check_cuda(cudaMemcpy(d_mass, h_mass.data(), bytes, cudaMemcpyHostToDevice), "copy mass failed");
        check_cuda(cudaMemcpy(d_charge, h_charge.data(), bytes, cudaMemcpyHostToDevice), "copy charge failed");
    });

    float *old_x = d_x_a, *old_y = d_y_a, *old_z = d_z_a;
    float *old_vx = d_vx_a, *old_vy = d_vy_a, *old_vz = d_vz_a;

    float *new_x = d_x_b, *new_y = d_y_b, *new_z = d_z_b;
    float *new_vx = d_vx_b, *new_vy = d_vy_b, *new_vz = d_vz_b;

    timings.kernel_ms = timer.measure([&] {
        for (int step = 0; step < num_steps; ++step) {
            simulate_step_soa_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
                old_x, old_y, old_z,
                old_vx, old_vy, old_vz,
                d_mass,
                d_charge,
                new_x, new_y, new_z,
                new_vx, new_vy, new_vz,
                n,
                force_mode
            );

            check_cuda(cudaGetLastError(), "SoA tiled kernel launch failed");
            check_cuda(cudaDeviceSynchronize(), "SoA tiled kernel execution failed");

            std::swap(old_x, new_x);
            std::swap(old_y, new_y);
            std::swap(old_z, new_z);

            std::swap(old_vx, new_vx);
            std::swap(old_vy, new_vy);
            std::swap(old_vz, new_vz);
        }
    });

    timings.d2h_ms = timer.measure([&] {
        check_cuda(cudaMemcpy(h_x.data(), old_x, bytes, cudaMemcpyDeviceToHost), "copy x back failed");
        check_cuda(cudaMemcpy(h_y.data(), old_y, bytes, cudaMemcpyDeviceToHost), "copy y back failed");
        check_cuda(cudaMemcpy(h_z.data(), old_z, bytes, cudaMemcpyDeviceToHost), "copy z back failed");

        check_cuda(cudaMemcpy(h_vx.data(), old_vx, bytes, cudaMemcpyDeviceToHost), "copy vx back failed");
        check_cuda(cudaMemcpy(h_vy.data(), old_vy, bytes, cudaMemcpyDeviceToHost), "copy vy back failed");
        check_cuda(cudaMemcpy(h_vz.data(), old_vz, bytes, cudaMemcpyDeviceToHost), "copy vz back failed");
    });

    for (int i = 0; i < n; ++i) {
        particles[i].x = h_x[i];
        particles[i].y = h_y[i];
        particles[i].z = h_z[i];

        particles[i].vx = h_vx[i];
        particles[i].vy = h_vy[i];
        particles[i].vz = h_vz[i];

        particles[i].mass = h_mass[i];
        particles[i].charge = h_charge[i];
    }

    if (print_timing) {
        print_timings(
            "Particle sim on GPU (SoA, tiled)",
            timings,
            blocks,
            threads_per_block,
            shared_mem_bytes
        );
    }
    if (timings_out != nullptr) {
        *timings_out = timings;
    }

    cudaFree(d_x_a);  cudaFree(d_y_a);  cudaFree(d_z_a);
    cudaFree(d_vx_a); cudaFree(d_vy_a); cudaFree(d_vz_a);

    cudaFree(d_x_b);  cudaFree(d_y_b);  cudaFree(d_z_b);
    cudaFree(d_vx_b); cudaFree(d_vy_b); cudaFree(d_vz_b);

    cudaFree(d_mass);
    cudaFree(d_charge);
}
