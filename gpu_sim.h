#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <vector>
#include "cpu_sim.h"

struct GpuTimings {
    float malloc_ms = 0.0f;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
};

void simulate_gpu_sqrt(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);
void simulate_gpu(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);
void simulate_gpu_tiled(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);
void simulate_gpu_tiled_sqrt(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);
void simulate_gpu_soa(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);
void simulate_gpu_soa_tiled(
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_timing = true,
    GpuTimings* timings_out = nullptr
);

#endif
