- CPU baseline simulation: `cpu`
- GPU implementations:
  - naive: `gpu_sqrt`, `gpu`
  - shared-memory tiled: `gpu_tiled`, `gpu_tiled_sqrt`
  - Structure of Arrays (SoA): `gpu_soa`
  - SoA + tiled: `gpu_soa_tiled`
- Force modes:
  - `gravity`
  - `electrostatic`
  - `combined`

## How To Run

1. Connect to a CUDA-capable GPU environment.
2. Build the project:

   ```bash
   make
   ```

3. Run the simulator:

   ```bash
   ./particle_sim <mode> <num_particles> <num_steps> <output_interval> <force_mode>
   ```

4. Example run:

   ```bash
   ./particle_sim gpu_tiled 10000 20 20 combined
   ```

5. Generate an animation from the CSV output:

   ```bash
   python animation/animate_particles.py animation/particles.csv simulation.gif
   ```

6. Run the benchmark suite:

   ```bash
   ./particle_sim benchmark 0 200 10 combined benchmark_results.csv
   ```

7. Plot benchmark results:

   ```bash
   python plot_benchmarks.py benchmark_results.csv benchmark_total_time.png benchmark_avg_gpu_time.png
   ```

## Dependencies

- CUDA Toolkit
- Standard C++ compiler
- Python with `pandas` and `matplotlib`

## Required Environment

- NVIDIA GPU with CUDA support

## Code Structure

- `gpu_sim.cu`: main CUDA kernels and GPU wrapper functions
- `gpu_sim.h`: GPU function declarations and timing structures
- `cpu_sim.cpp`: CPU baseline implementation and CSV output
- `cpu_sim.h`: shared particle definitions and simulation constants
- `main.cpp`: entry point, mode selection, animation output, and benchmarking
- `animation/animate_particles.py`: generates GIF animations from particle CSV output
- `plot_benchmarks.py`: plots benchmark CSV results with matplotlib