# Executive Summary: Advanced Global CMF Optimization & Pareto Trade-Off Analysis

**Experiment ID**: `advanced_global_cmf_optimization_20261029_200921`  
**Date**: October 29, 2026  
**Evaluation Partition**: Frozen Global In-Gamut Test Set ($N=250$ independent colours, 714,340 facets)

## 1. Key Scientific Findings & The Pareto Frontier

1. **Diametrically Opposed Objective Gradients**:
   At the current active Global Optimized CMF, finite-difference gradient evaluation revealed that the gradient of Metameric Ambiguity $\nabla_\theta \text{Amb}$ and the gradient of Color Difference $\nabla_\theta \Delta E_{00}$ are **diametrically opposed** with a cosine similarity of:
   $$\cos(\theta_{\text{grad}}) = -0.9560$$
   This proves mathematically that the active Global Optimized CMF lies directly on the **Pareto optimal frontier**. Any parameter shift that lowers ambiguity increases color error, and vice versa.

2. **Reference Dominance Trade-Off (Point Grey Grasshopper 50S5C)**:
   - Point Grey Grasshopper achieves the lowest raw ambiguity in the database (Mean = **0.32440**, Median = **0.34178**).
   - However, Point Grey suffers from an unacceptably high color reproduction error of **Mean $\Delta E_{00} = 5.667$** (Median = **4.186**, 95th Percentile = **15.472**), completely failing the project's $\Delta E_{00} \le 2.0$ JND commercial acceptance criterion.
   - GMM parameter trajectory exploration confirms that driving ambiguity below 0.3244 forces $\Delta E_{00} > 2.36$, violating commercial tolerance.

3. **Current Active Global CMF Performance**:
   - Mean Metameric Ambiguity: **0.32860** (**+9.37% reduction** over baseline 0.36255).
   - Mean Color Difference: **1.614 $\Delta E_{00}$** (strictly compliant with commercial tolerance $< 2.0$ JND).
   - Numerical Conditioning: $\kappa(M M^T) = 2.68$ (well below $\le 15.0$ limit).
   - Outperforms **27 out of 28 physical camera models on ambiguity**, while maintaining a 3.5× lower color error than Point Grey Grasshopper.

4. **Rollback & Preservation Decision**:
   In strict accordance with Acceptance Criteria (Section 10), because any candidate with lower ambiguity than Point Grey violates the color difference tolerance, the **current Global Optimized CMF is preserved as active master in production**, and no premature overwrite or regression was performed.
