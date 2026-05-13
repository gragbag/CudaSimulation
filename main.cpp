#include "cpu_sim.h"
#include "gpu_sim.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int parse_force_mode(const std::string& force_name) {
    if (force_name == "gravity") {
        return 0;
    }
    if (force_name == "electrostatic") {
        return 1;
    }
    if (force_name == "combined") {
        return 2;
    }

    std::cerr << "Invalid force mode. Use: gravity, electrostatic, or combined.\n";
    std::exit(1);
}

ForceMode parse_cpu_force_mode(const std::string& force_name) {
    if (force_name == "gravity") {
        return ForceMode::Gravity;
    }
    if (force_name == "electrostatic") {
        return ForceMode::Electrostatic;
    }
    if (force_name == "combined") {
        return ForceMode::Combined;
    }

    std::cerr << "Invalid force mode. Use: gravity, electrostatic, or combined.\n";
    std::exit(1);
}

struct RunMetrics {
    double total_time_s = 0.0;
    GpuTimings avg_chunk_timings;
    bool has_gpu_timings = false;
};

void print_gpu_chunk_average_timings(
    const std::string& mode,
    const GpuTimings& avg_timings,
    int chunks
) {
    if (chunks <= 0) {
        return;
    }

    std::cout << "\nAverage GPU chunk timings (" << mode << ")\n";
    std::cout << "    chunks:          " << chunks << "\n";
    std::cout << "    cudaMalloc:      " << avg_timings.malloc_ms / 1000.0f << " s\n";
    std::cout << "    cudaMemcpy H2D:  " << avg_timings.h2d_ms / 1000.0f << " s\n";
    std::cout << "    kernel:          " << avg_timings.kernel_ms / 1000.0f << " s\n";
    std::cout << "    cudaMemcpy D2H:  " << avg_timings.d2h_ms / 1000.0f << " s\n";
}

bool is_gpu_mode(const std::string& mode) {
    return mode == "gpu_sqrt" ||
           mode == "gpu" ||
           mode == "gpu_tiled_sqrt" ||
           mode == "gpu_tiled" ||
           mode == "gpu_soa" ||
           mode == "gpu_soa_tiled";
}

RunMetrics run_cpu_mode(
    std::vector<Particle>& particles,
    int num_steps,
    int output_interval,
    ForceMode force_mode,
    const std::string& output_file,
    bool write_csv
) {
    RunMetrics metrics;
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= num_steps; ++step) {
        simulate_step(particles, force_mode);

        if (write_csv && step % output_interval == 0) {
            write_particles_to_csv(output_file, particles, step, false);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.total_time_s = std::chrono::duration<double>(end - start).count();
    return metrics;
}

RunMetrics run_gpu_mode(
    const std::string& mode,
    std::vector<Particle>& particles,
    int num_steps,
    int output_interval,
    int force_mode,
    const std::string& output_file,
    bool write_csv,
    bool print_avg_summary
) {
    RunMetrics metrics;

    auto run_steps = [&](int steps) {
        GpuTimings timings;

        if (mode == "gpu_sqrt") {
            simulate_gpu_sqrt(particles, steps, force_mode, false, &timings);
        } else if (mode == "gpu") {
            simulate_gpu(particles, steps, force_mode, false, &timings);
        } else if (mode == "gpu_tiled_sqrt") {
            simulate_gpu_tiled_sqrt(particles, steps, force_mode, false, &timings);
        } else if (mode == "gpu_tiled") {
            simulate_gpu_tiled(particles, steps, force_mode, false, &timings);
        } else if (mode == "gpu_soa") {
            simulate_gpu_soa(particles, steps, force_mode, false, &timings);
        } else if (mode == "gpu_soa_tiled") {
            simulate_gpu_soa_tiled(particles, steps, force_mode, false, &timings);
        }

        return timings;
    };

    auto start = std::chrono::high_resolution_clock::now();
    int completed_steps = 0;
    int chunks = 0;
    GpuTimings total_timings;

    while (completed_steps < num_steps) {
        int chunk_steps = std::min(output_interval, num_steps - completed_steps);
        GpuTimings chunk_timings = run_steps(chunk_steps);

        total_timings.malloc_ms += chunk_timings.malloc_ms;
        total_timings.h2d_ms += chunk_timings.h2d_ms;
        total_timings.kernel_ms += chunk_timings.kernel_ms;
        total_timings.d2h_ms += chunk_timings.d2h_ms;

        completed_steps += chunk_steps;
        ++chunks;
        if (write_csv) {
            write_particles_to_csv(output_file, particles, completed_steps, false);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.total_time_s = std::chrono::duration<double>(end - start).count();
    metrics.has_gpu_timings = (chunks > 0);

    if (chunks > 0) {
        metrics.avg_chunk_timings.malloc_ms = total_timings.malloc_ms / chunks;
        metrics.avg_chunk_timings.h2d_ms = total_timings.h2d_ms / chunks;
        metrics.avg_chunk_timings.kernel_ms = total_timings.kernel_ms / chunks;
        metrics.avg_chunk_timings.d2h_ms = total_timings.d2h_ms / chunks;
    }

    if (print_avg_summary && chunks > 0) {
        print_gpu_chunk_average_timings(mode, metrics.avg_chunk_timings, chunks);
    }

    return metrics;
}

RunMetrics run_gpu_mode_once(
    const std::string& mode,
    std::vector<Particle>& particles,
    int num_steps,
    int force_mode,
    bool print_summary
) {
    RunMetrics metrics;
    GpuTimings timings;

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "gpu_sqrt") {
        simulate_gpu_sqrt(particles, num_steps, force_mode, false, &timings);
    } else if (mode == "gpu") {
        simulate_gpu(particles, num_steps, force_mode, false, &timings);
    } else if (mode == "gpu_tiled_sqrt") {
        simulate_gpu_tiled_sqrt(particles, num_steps, force_mode, false, &timings);
    } else if (mode == "gpu_tiled") {
        simulate_gpu_tiled(particles, num_steps, force_mode, false, &timings);
    } else if (mode == "gpu_soa") {
        simulate_gpu_soa(particles, num_steps, force_mode, false, &timings);
    } else if (mode == "gpu_soa_tiled") {
        simulate_gpu_soa_tiled(particles, num_steps, force_mode, false, &timings);
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.total_time_s = std::chrono::duration<double>(end - start).count();
    metrics.has_gpu_timings = true;
    metrics.avg_chunk_timings = timings;

    if (print_summary) {
        print_gpu_chunk_average_timings(mode, metrics.avg_chunk_timings, 1);
    }

    return metrics;
}

void write_benchmark_header(std::ofstream& file) {
    file << "mode,num_particles,num_steps,output_interval,force_mode,total_time_s,avg_step_s,"
         << "avg_gpu_malloc_s,avg_gpu_h2d_s,avg_gpu_kernel_s,avg_gpu_d2h_s\n";
}

void write_benchmark_row(
    std::ofstream& file,
    const std::string& mode,
    int num_particles,
    int num_steps,
    int output_interval,
    const std::string& force_name,
    const RunMetrics& metrics
) {
    file << mode << ","
         << num_particles << ","
         << num_steps << ","
         << output_interval << ","
         << force_name << ","
         << metrics.total_time_s << ","
         << (metrics.total_time_s / num_steps) << ",";

    if (metrics.has_gpu_timings) {
        file << metrics.avg_chunk_timings.malloc_ms / 1000.0f << ","
             << metrics.avg_chunk_timings.h2d_ms / 1000.0f << ","
             << metrics.avg_chunk_timings.kernel_ms / 1000.0f << ","
             << metrics.avg_chunk_timings.d2h_ms / 1000.0f;
    } else {
        file << "0,0,0,0";
    }

    file << "\n";
}

void run_benchmark_suite(
    int num_steps,
    int output_interval,
    int gpu_force_mode,
    ForceMode cpu_force_mode,
    const std::string& force_name,
    unsigned int seed,
    const std::string& benchmark_file
) {
    const std::vector<int> particle_counts = {10, 100, 1000, 10000, 50000, 100000};
    const std::vector<std::string> modes = {
        "cpu",
        "gpu_sqrt",
        "gpu",
        "gpu_tiled_sqrt",
        "gpu_tiled",
        "gpu_soa",
        "gpu_soa_tiled"
    };

    std::ofstream file(benchmark_file, std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open benchmark CSV file: " + benchmark_file);
    }

    write_benchmark_header(file);

    for (int num_particles : particle_counts) {
        std::vector<Particle> baseline_particles(num_particles);
        initialize_particles(baseline_particles, seed);

        for (const std::string& mode : modes) {
            if (mode == "cpu" && num_particles >= 50000) {
                std::cout << "Skipping benchmark: mode=cpu, particles="
                          << num_particles << " (too slow)\n";
                continue;
            }

            std::vector<Particle> particles = baseline_particles;
            RunMetrics metrics;

            if (mode == "cpu") {
                metrics = run_cpu_mode(
                    particles,
                    num_steps,
                    output_interval,
                    cpu_force_mode,
                    "",
                    false
                );
            } else {
                metrics = run_gpu_mode_once(
                    mode,
                    particles,
                    num_steps,
                    gpu_force_mode,
                    false
                );
            }

            write_benchmark_row(
                file,
                mode,
                num_particles,
                num_steps,
                output_interval,
                force_name,
                metrics
            );

            std::cout << "Benchmark complete: mode=" << mode
                      << ", particles=" << num_particles
                      << ", total_time=" << metrics.total_time_s << " s\n";
        }
    }

    std::cout << "\nSaved benchmark results to " << benchmark_file << "\n";
}

int main(int argc, char* argv[]) {
    std::string mode = "cpu";
    int num_particles = 1000;
    int num_steps = 100;
    int output_interval = 1;
    std::string force_name = "gravity";

    unsigned int seed = 42;
    std::string output_file = "animation/particles.csv";
    std::string benchmark_file = "benchmark_results.csv";

    if (argc >= 2) {
        mode = argv[1];
    }
    if (argc >= 3) {
        num_particles = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        num_steps = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        output_interval = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        force_name = argv[5];
    }
    if (argc >= 7) {
        if (mode == "benchmark") {
            benchmark_file = argv[6];
        } else {
            output_file = argv[6];
        }
    }

    int gpu_force_mode = parse_force_mode(force_name);
    ForceMode cpu_force_mode = parse_cpu_force_mode(force_name);

    if (mode == "benchmark") {
        run_benchmark_suite(
            num_steps,
            output_interval,
            gpu_force_mode,
            cpu_force_mode,
            force_name,
            seed,
            benchmark_file
        );
        return 0;
    }

    std::vector<Particle> particles(num_particles);
    initialize_particles(particles, seed);

    write_particles_to_csv(output_file, particles, 0, true);

    RunMetrics metrics;
    if (mode == "cpu") {
        metrics = run_cpu_mode(
            particles,
            num_steps,
            output_interval,
            cpu_force_mode,
            output_file,
            true
        );
    } else if (is_gpu_mode(mode)) {
        metrics = run_gpu_mode(
            mode,
            particles,
            num_steps,
            output_interval,
            gpu_force_mode,
            output_file,
            true,
            true
        );
    } else {
        std::cerr << "Invalid mode. Use one of:\n"
                  << "  cpu\n"
                  << "  gpu\n"
                  << "  gpu_tiled_sqrt\n"
                  << "  gpu_tiled\n"
                  << "  gpu_soa\n"
                  << "  gpu_soa_tiled\n"
                  << "  benchmark\n";
        return 1;
    }

    std::cout << "Mode:            " << mode << "\n";
    std::cout << "Force mode:      " << force_name << "\n";
    std::cout << "Particles:       " << num_particles << "\n";
    std::cout << "Steps:           " << num_steps << "\n";
    std::cout << "Output interval: " << output_interval << "\n";
    std::cout << "CSV file:        " << output_file << "\n";
    std::cout << "Total time:      " << metrics.total_time_s << " s\n";
    std::cout << "Avg/step:        " << (metrics.total_time_s / num_steps) << " s\n\n";

    std::cout << "Final positions of first 5 particles:\n";
    for (int i = 0; i < std::min(5, num_particles); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << "Particle " << i << ": ("
                  << particles[i].x << ", "
                  << particles[i].y << ", "
                  << particles[i].z << "), charge = "
                  << particles[i].charge << "\n";
    }

    return 0;
}
