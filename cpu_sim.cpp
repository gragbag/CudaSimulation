#include "cpu_sim.h"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

void initialize_particles(std::vector<Particle>& particles, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 10.0f);
    std::uniform_real_distribution<float> charge_dist(-1.0f, 1.0f);

    for (auto& p : particles) {
        p.x = pos_dist(rng);
        p.y = pos_dist(rng);
        p.z = pos_dist(rng);

        p.vx = vel_dist(rng);
        p.vy = vel_dist(rng);
        p.vz = vel_dist(rng);

        p.mass = mass_dist(rng);

        //p.charge = charge_dist(rng);
        p.charge = (charge_dist(rng) < 0.0f) ? -1.0f : 1.0f;
    }
}

void simulate_step(std::vector<Particle>& particles, ForceMode mode) {
    const size_t n = particles.size();

    std::vector<float> fx(n, 0.0f);
    std::vector<float> fy(n, 0.0f);
    std::vector<float> fz(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        float xi = particles[i].x;
        float yi = particles[i].y;
        float zi = particles[i].z;

        float mi = particles[i].mass;
        float qi = particles[i].charge;

        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }

            float dx = particles[j].x - xi;
            float dy = particles[j].y - yi;
            float dz = particles[j].z - zi;

            float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;

            float inv_dist = 1.0f / std::sqrt(dist_sqr);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float scale = 0.0f;

            if (mode == ForceMode::Gravity || mode == ForceMode::Combined) {
                float mj = particles[j].mass;
                scale += G * mi * mj * inv_dist3;
            }

            if (mode == ForceMode::Electrostatic || mode == ForceMode::Combined) {
                float qj = particles[j].charge;

                // Negative sign because same charges repel and opposite charges attract.
                scale += -K_E * qi * qj * inv_dist3;
            }

            fx[i] += scale * dx;
            fy[i] += scale * dy;
            fz[i] += scale * dz;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        float ax = fx[i] / particles[i].mass;
        float ay = fy[i] / particles[i].mass;
        float az = fz[i] / particles[i].mass;

        particles[i].vx += ax * DT;
        particles[i].vy += ay * DT;
        particles[i].vz += az * DT;

        particles[i].x += particles[i].vx * DT;
        particles[i].y += particles[i].vy * DT;
        particles[i].z += particles[i].vz * DT;
    }
}

void write_particles_to_csv(
    const std::string& filename,
    const std::vector<Particle>& particles,
    int step,
    bool write_header
) {
    std::ofstream file;

    if (write_header) {
        file.open(filename, std::ios::out);
    } else {
        file.open(filename, std::ios::app);
    }

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }

    if (write_header) {
        file << "step,id,x,y,z,mass,charge\n";
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        file << step << ","
             << i << ","
             << particles[i].x << ","
             << particles[i].y << ","
             << particles[i].z << ","
             << particles[i].mass << ","
             << particles[i].charge << "\n";
    }
}
