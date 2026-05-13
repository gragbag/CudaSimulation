# GPU Particle Simulation (CUDA)

This project implements a 3D particle simulation with GPU acceleration using CUDA. It includes multiple optimized versions and supports different force models (gravity, electrostatic, and combined).

- CPU baseline simulation (`cpu`)
- GPU implementations:
  - naive (`gpu_sqrt`, `gpu`)
  - shared-memory tiled (`gpu_tiled`)
  - Structure of Arrays (SoA) (`gpu_soa`)
  - SoA + tiled (`gpu_soa_tiled`)
- Force modes:
  - gravity
  - electrostatic
  - combined

How to Run:
    1. Connect to GPU
    2. run 'make' to compile
    3. ./particle_sim <mode> <num_particles> <num_steps> <output_interval> <force_mode>
    example: ./particle_sim gpu_tiled 10000 20 20 combined

Can also get the animation for the particles with:
    python animation/animate_particles.py animation/particles.csv simulation.gif

Dependencies
    CUDA Toolkit
    Standard C++ compiler

    Python with pandas and matplotlib

Required Environment
    NVIDIA GPU with CUDA support

Code Structure
    - gpu_sim.cu
        - Main CUDA kernels (naive, tiled, SoA, SoA+tiled)
    - cpu_sim.cpp
        - CPU baseline implementation
    - main.cpp
        -Entry point, mode selection
    - animate_particles.py
        - Visualization and demo, generates GIF outputs