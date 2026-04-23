# **PhaseField_PINN-MT**

*Physics-informed neural networks for forward simulation and inverse parameter estimation in 2D martensitic phase transformations.*

---

## **Overview**

This repository contains the research code accompanying the paper:

**Asfandyar Khan** and **Mahmood Mamivand**  
*Physics-Informed Neural Networks for Martensitic Transformation: Toward Morphology-Based Material Parameters Estimation*  
**Physica Scripta**  
**DOI:** `10.1088/1402-4896/ae5fe5`

This repository provides implementations of **physics-informed neural networks (PINNs)** for a **2D martensitic phase-field model** governed by coupled **time-dependent Ginzburg-Landau (TDGL)** and **mechanical equilibrium equations**.

The framework includes:

- **Forward PINN**: predicts *spatiotemporal microstructure evolution* from prescribed initial conditions  
- **Inverse PINN**: performs *morphology-based estimation* of the gradient energy coefficient from *sparse, partial, and noisy observations*

---

## **Scientific Scope**

Martensitic transformations are modeled here using a coupled **phase-field + mechanics** formulation. In the associated study, the PINN framework is used to:

- solve the forward **TDGL–mechanical equilibrium** system
- estimate the **gradient energy coefficient** from microstructure observations
- evaluate performance under **sparse sampling**, **partial-domain observations**, and **noisy data**

This work is intended to support **reproducible research**, **open scientific dissemination**, and further development of **physics-informed machine learning methods in computational materials science**.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt


## **Repository Structure**

```text
PhaseField_PINN-MT/
├── Forward_PINN/        # Forward PINN code and related files
├── Inverse_PINN/        # Inverse PINN code and related files
├── README.md            # Project overview and usage instructions
└── ...
