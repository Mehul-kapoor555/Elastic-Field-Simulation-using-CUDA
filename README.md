# Elastic Field Simulation using CUDA and Spectral Methods

## Overview

This project implements a **GPU-accelerated elastic field simulation** using CUDA, cuFFT, and Thrust.

The simulation models the interaction between polarization, strain, stress, and resulting elastic forces in a 3D grid using a **Fourier-space (spectral) approach**.

It represents a full computational pipeline for solving mechanical equilibrium in materials using high-performance parallel computing.

---

## Key Features

* CUDA-based parallel computation
* FFT-based spectral solver using cuFFT
* Full elastic field pipeline:

  * Polarization → Strain → Stress → Elastic Forces
* 3D grid simulation
* High-performance implementation using Thrust

---

## Computational Pipeline

The simulation follows these steps:

1. **Polarization Initialization**
   Initializes polarization field across a 3D grid

2. **Spontaneous Strain Calculation**
   Computes strain tensor components from polarization

3. **Fourier Transform (cuFFT)**
   Converts strain fields into frequency domain

4. **Mechanical Equilibrium Solver**
   Solves linear systems in Fourier space

5. **Displacement Field Computation**
   Computes displacement via matrix inversion

6. **Strain & Stress Calculation**
   Derives total strain and stress tensors

7. **Elastic Force Computation**
   Computes forces from stress divergence in Fourier space

8. **Inverse FFT**
   Converts results back to real space

---

## Core Equations

The simulation is based on elastic equilibrium and Fourier-space formulation.

---

### 1. Spontaneous Strain from Polarization

ε⁰_ij = Q_ijkl * P_k * P_l

where:
- P = polarization vector  
- Q = electrostrictive coefficients  

---

### 2. Elastic Stress-Strain Relation

σ_ij = c_ijkl (ε_kl - ε⁰_kl)

where:
- σ_ij = stress tensor  
- ε_kl = total strain  
- ε⁰_kl = spontaneous strain  
- c_ijkl = elastic stiffness tensor  

---

### 3. Mechanical Equilibrium (Fourier Space)

∂σ_ij / ∂x_j = 0  

In Fourier space:  

k_j * σ_ij = 0  

where:
- σ_ij = stress tensor  
- k_j = wave vector in Fourier space  
- This condition enforces force balance in the system  

---

### 4. Displacement-Strain Relation

ε_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)

where:
- u_i = displacement field  
- ε_ij = strain tensor derived from displacement  

---

### 5. Fourier Derivative Property

∂/∂x → i * k

where:
- i = imaginary unit  
- k = wave vector  
- Converts spatial derivatives into algebraic operations in Fourier space  

**The system is solved in Fourier space for computational efficiency.**

---

## Numerical Approach

- The domain is discretized into a 3D grid  
- Real-space fields are transformed to Fourier space using cuFFT  
- Differential equations are converted into algebraic equations  
- Linear systems are solved independently for each k-point  
- Inverse FFT is used to reconstruct real-space fields  

This approach significantly improves computational efficiency for large-scale simulations.

---

## Technologies Used

* CUDA
* cuFFT
* Thrust
* C++

---

## Why GPU?

This problem involves:

* Large 3D grids
* Tensor operations
* FFT computations

Making it highly suitable for **GPU acceleration**

---

## Project Structure

```
src/
 └── simulation.cu
```

---

## How to Compile

```bash
nvcc src/simulation.cu -lcufft -o simulation
```

---

## How to Run

```bash
./simulation
```

---

## Implementation Highlights

- Efficient use of GPU memory via Thrust device vectors  
- Parallel computation using transform-based operations  
- FFT-based spectral solving instead of finite differences  
- Decoupled pipeline allowing independent computation at each k-point

---

## References

1. Y. L. Li et al., "Phase-field modeling of ferroelectric domain structures," Acta Materialia (2002)

This project is inspired by classical phase-field and elastic field modeling approaches used in ferroelectric materials research.

The research paper is uploaded as a pdf in `docs`.

---

## Note

This project requires:

* NVIDIA GPU
* CUDA Toolkit installed

---

## Key Takeaways

* Implementation of physics-based simulation on GPU
* Use of spectral (FFT-based) methods for solving PDEs
* Efficient parallelization of multi-stage computational pipeline
* Integration of physics, mathematics, and high-performance computing

---

## Future Improvements

* Add visualization of elastic fields
* Optimize memory usage
* Extend to larger grid sizes
* Integrate with external plotting tools

---

## Author

Mehul Kapoor
