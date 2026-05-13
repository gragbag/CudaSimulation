#ifndef CPU_SIM_H
#define CPU_SIM_H

#include <string>
#include <vector>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    float charge;
};

enum class ForceMode {
    Gravity,
    Electrostatic,
    Combined,
};

constexpr float G = 10.0f;
constexpr float K_E = 50.0f;
constexpr float SOFTENING = 1e-5f;
constexpr float DT = 0.05f;

void initialize_particles(std::vector<Particle>& particles, unsigned int seed);
void simulate_step(std::vector<Particle>& particles, ForceMode mode);
void write_particles_to_csv(
    const std::string& filename,
    const std::vector<Particle>& particles,
    int step,
    bool write_header
);

#endif