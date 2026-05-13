NVCC = nvcc
NVCCFLAGS = -O2 -std=c++17

all: particle_sim

particle_sim: main.cpp cpu_sim.cpp gpu_sim.cu cpu_sim.h gpu_sim.h
	$(NVCC) $(NVCCFLAGS) main.cpp cpu_sim.cpp gpu_sim.cu -o particle_sim

clean:
	rm -f particle_sim particles.csv