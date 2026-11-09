# Optimization Convergence Dynamics and Trajectories

## 1. Algorithm Settings
- **Optimizer:** L-BFGS-B with bound constraints
- **Tolerances:** ftol = 1e-6, gtol = 1e-5, maxiter = 45
- **Convergence Rate:** 100% solver feasibility across all 13 runs (8 converged by ftol/gtol, 5 reached iteration budget at flat valley).

## 2. Dynamics
- Monotonic decrease in total objective {\\text{total}}$ observed across all trajectories.
- Primary ambiguity reduction occurs within the first 10-15 iterations.
- Colorimetric distortion ($\\Delta E_{00}$) was strictly constrained to $\\le 1.84$ JND across all runs.
