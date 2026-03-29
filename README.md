# Elastic Field Simulation using Phase-Field Modeling

## Overview

This project implements a computational model for simulating **elastic field interactions in ferroelectric thin films** using a **phase-field approach**.

The model captures the evolution of domain structures under elastic and thermodynamic constraints, inspired by classical research in ferroelectric materials and microstructure evolution.

---

## Background

Ferroelectric materials exhibit spontaneous polarization that evolves into domain structures due to competing energy contributions:

* Bulk (Landau) free energy
* Elastic strain energy
* Domain wall (gradient) energy

The system evolves to minimize total free energy, governed by the **Time-Dependent Ginzburg–Landau (TDGL) equation**:

[
\frac{\partial P_i}{\partial t} = -L \frac{\delta F}{\delta P_i}
]

where:

* (P_i) is the polarization field
* (F) is the total free energy
* (L) is a kinetic coefficient

---

## Model Components

The simulation incorporates:

### 1. Landau Free Energy

Polynomial expansion describing phase transition behavior.

### 2. Gradient Energy

Captures domain wall formation via spatial variation of polarization.

### 3. Elastic Energy

Accounts for strain induced by polarization:

[
\sigma_{ij} = c_{ijkl}(\epsilon_{kl} - \epsilon^0_{kl})
]

This introduces coupling between mechanical deformation and polarization fields.

---

## Key Features

* 3D phase-field formulation
* Coupled electro-mechanical modeling
* Elastic field computation under substrate constraints
* Time evolution of domain structures using TDGL dynamics

---

## Implementation Details

* Language: MATLAB
* Approach: Numerical simulation using discretized field equations
* Solver: Time-stepping evolution based on energy minimization

The implementation focuses on **conceptual correctness and physical modeling**, rather than optimized large-scale computation.

---

## Limitations

* No GPU acceleration (CPU-based simulation)
* Large-scale simulations may be computationally expensive
* Visualization not included due to hardware constraints

---

## 📚 References

This implementation is inspired by foundational work in phase-field modeling of ferroelectric materials:

* Li et al., *Acta Materialia (2002)* — Ferroelectric domain evolution using phase-field methods

---

## Future Improvements

* Add visualization of domain structures
* Optimize computation using parallel processing / GPU
* Extend model to include electrostatic interactions
* Export simulation data for external plotting (Python/Matplotlib)

---

## Key Takeaway

This project demonstrates the implementation of **physics-driven simulation models**, combining:

* Partial differential equations
* Numerical methods
* Materials science concepts

It reflects the ability to translate **theoretical research models into working computational code**.

---

## Project Structure

```
elastic-field-project/
│
├── src/
│   └── simulation.m
│
├── README.md
├── LICENSE
├── .gitignore
```

---

## Author

Mehul Kapoor
