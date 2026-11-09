# Feasible Spectral Gamut Coverage Analysis

## 1. Gamut Polytope Specification
The empirical spectral reflectance polytope $P_{\text{spec}}$ in $\mathbb{R}^8$ is bounded by 714,340 linear inequalities:
$$A_{\text{red}} \mathbf{z} \le \mathbf{b}_{\text{red}}$$
Standard RGB triplets $[0, 255]^3$ do not represent valid spectral reflectances. Physical reflectance spectra must satisfy $0 \le R(\lambda) \le 1$ and natural smoothness constraints.

## 2. Numerical Discretization of Continuous Gamut
Because the continuous feasible gamut contains infinitely many colours, we constructed a well-distributed numerical representation ($N=500$) stratified across:
- **12 Hue Sectors (30° bins)**: Red, Orange, Yellow-Orange, Yellow-Green, Green, Teal, Cyan, Sky Blue, Blue, Violet, Magenta, Purple.
- **Radial Depth Tiers**: Deep interior ($r \in [0.40, 0.65]$), Mid interior ($r \in [0.70, 0.88]$), and Gamut boundary ($r \in [0.94, 0.985]$).
- **Gamut Boundary Vertices**: $N=75$ extreme surface vertices ($r \in [0.96, 0.99]$).
- **Near-Neutral Colours**: $N=50$ points with chromaticity $s < 0.05$.

## 3. Evaluation Partitions
- **Optimization Set (Train)**: $N=150$ points
- **Model Selection Set (Validation)**: $N=100$ points
- **Frozen Evaluation Set (Test)**: $N=250$ points
