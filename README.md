# PhaseField-PINN-MT

## Overview
This repository contains the code accompanying the manuscript:

**Physics-Informed Neural Networks for Martensitic Transformation: Toward Morphology-Based Material Parameters Estimation**

We developed a physics-informed neural networks (PINN) for a 2D martensitic phase-field model governed by a **coupled time-dependent Ginzburg–Landau (TDGL)** and **mechanical equilibrium** equations. The repository includes:

- **Forward PINN**: solves the coupled TDGL + mechanical equilibrium system to predict spatiotemporal microstructure evolution from localized perturbations.
- **Inverse PINN**: performs **morphology-based parameter estimation** of the **gradient energy coefficient β** from sparse, partial, and noisy observations.
