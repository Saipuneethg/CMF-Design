import numpy as np


class HitAndRunSampler:
    """
    Constrained Hit-and-Run Markov Chain Monte Carlo sampler for the bounded
    reduced-space polytope:
    
        G @ alpha <= h
        
    where alpha in R^{d_f}, G in R^{M x d_f}, and h in R^M.
    
    Generates numerical representations of the continuous feasible fiber.
    Supports user-configured sample count N, burn-in, thinning, progress callbacks,
    reproducible random seeds, and rigorous numerical safeguards.
    """
    def __init__(self, G, h, tol=1e-7):
        self.G = np.asarray(G, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.tol = tol
        self.dim = self.G.shape[1]
        self.num_constraints = self.G.shape[0]

    def is_feasible(self, alpha):
        """Checks whether alpha satisfies G @ alpha <= h within tolerance."""
        violations = self.G @ alpha - self.h
        return np.all(violations <= self.tol), np.max(violations)

    def sample_chain(
        self,
        num_samples=1000,
        alpha_init=None,
        burn_in=50,
        thinning=2,
        seed=42,
        resync_interval=50,
        progress_callback=None
    ):
        """
        Samples a single Markov chain in the reduced feasible polytope G @ alpha <= h.
        
        Guarantees returning exactly num_samples points.
        
        Parameters:
        - num_samples: Number of retained samples (N).
        - alpha_init: Feasible initial point in R^{d_f}. If None, defaults to zeros.
        - burn_in: Number of initial burn-in iterations discarded.
        - thinning: Step interval between saved samples.
        - seed: Integer random seed for reproducibility.
        - resync_interval: Steps between full slack re-evaluations to avoid numerical drift.
        - progress_callback: Optional callable(current_samples, total_samples).
        
        Returns:
        - samples: Array of shape (num_samples, dim).
        - diagnostics: Dict with sampling health metrics.
        """
        rng = np.random.RandomState(seed)
        num_samples = int(num_samples)
        burn_in = int(burn_in)
        thinning = int(thinning)
        
        if alpha_init is None:
            alpha = np.zeros(self.dim, dtype=np.float64)
        else:
            alpha = np.array(alpha_init, dtype=np.float64).flatten()
            if len(alpha) != self.dim:
                raise ValueError(f"alpha_init length {len(alpha)} != dimension {self.dim}")

        feasible, max_viol = self.is_feasible(alpha)
        if not feasible:
            raise ValueError(
                f"Initial point is infeasible: max constraint violation = {max_viol:.4e} > {self.tol}"
            )

        slack = self.h - self.G @ alpha
        slack = np.maximum(slack, 0.0)

        samples = []
        accepted_moves = 0
        degenerate_dirs = 0
        attempts = 0
        max_attempts = max(2000, (burn_in + num_samples * thinning) * 10)

        # 1. Burn-in phase
        burn_in_done = 0
        while burn_in_done < burn_in and attempts < max_attempts:
            attempts += 1
            direction = rng.randn(self.dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                degenerate_dirs += 1
                continue
            direction /= norm

            G_dir = self.G @ direction
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = slack / G_dir
                pos_mask = G_dir > 1e-9
                neg_mask = G_dir < -1e-9
                t_max = np.min(np.where(pos_mask, ratios, np.inf))
                t_min = np.max(np.where(neg_mask, ratios, -np.inf))

            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                degenerate_dirs += 1
                continue

            interval_len = t_max - t_min
            eps = min(1e-7, 1e-3 * interval_len)
            if interval_len > 2 * eps:
                t = rng.uniform(t_min + eps, t_max - eps)
            else:
                t = rng.uniform(t_min, t_max)

            alpha = alpha + t * direction
            slack = slack - t * G_dir
            accepted_moves += 1
            burn_in_done += 1

            if burn_in_done % resync_interval == 0:
                slack = self.h - self.G @ alpha
                slack = np.maximum(slack, 0.0)

        # 2. Sampling phase with thinning
        thin_count = 0
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            direction = rng.randn(self.dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                degenerate_dirs += 1
                continue
            direction /= norm

            G_dir = self.G @ direction
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = slack / G_dir
                pos_mask = G_dir > 1e-9
                neg_mask = G_dir < -1e-9
                t_max = np.min(np.where(pos_mask, ratios, np.inf))
                t_min = np.max(np.where(neg_mask, ratios, -np.inf))

            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                degenerate_dirs += 1
                continue

            interval_len = t_max - t_min
            eps = min(1e-7, 1e-3 * interval_len)
            if interval_len > 2 * eps:
                t = rng.uniform(t_min + eps, t_max - eps)
            else:
                t = rng.uniform(t_min, t_max)

            alpha = alpha + t * direction
            slack = slack - t * G_dir
            accepted_moves += 1
            thin_count += 1

            if accepted_moves % resync_interval == 0:
                slack = self.h - self.G @ alpha
                slack = np.maximum(slack, 0.0)

            if thin_count >= thinning:
                samples.append(alpha.copy())
                thin_count = 0
                if progress_callback:
                    progress_callback(len(samples), num_samples)

        samples_arr = np.array(samples)
        
        # Polytope violations check for retained points (memory-safe chunking)
        if len(samples_arr) > 0:
            chunk_size = 25
            all_viols = []
            for i in range(0, len(samples_arr), chunk_size):
                s_chunk = samples_arr[i : i + chunk_size]
                slacks = s_chunk @ self.G.T
                slacks -= self.h
                all_viols.extend(np.max(slacks, axis=1))
                del slacks
            all_viols = np.array(all_viols)
            max_viol_retained = float(np.max(all_viols))
            valid_retained = int(np.sum(all_viols <= self.tol))
        else:
            max_viol_retained = float("nan")
            valid_retained = 0

        diagnostics = {
            "total_attempts": attempts,
            "burn_in_steps": burn_in_done,
            "accepted_moves": accepted_moves,
            "acceptance_rate": accepted_moves / max(attempts, 1),
            "degenerate_directions": degenerate_dirs,
            "samples_collected": len(samples_arr),
            "max_violation_retained": max_viol_retained,
            "valid_samples_retained": valid_retained,
            "seed": seed,
            "n_requested": num_samples,
            "burn_in": burn_in,
            "thinning": thinning
        }

        return samples_arr, diagnostics

    def sample_multiple_chains(
        self,
        num_chains=3,
        num_samples_per_chain=1000,
        initial_points=None,
        burn_in=50,
        thinning=2,
        base_seed=42
    ):
        """
        Runs multiple independently seeded Hit-and-Run chains.
        """
        chains = []
        diagnostics_list = []

        for c_idx in range(num_chains):
            chain_seed = base_seed + c_idx * 1000 + 7
            init_pt = None
            if initial_points is not None and c_idx < len(initial_points):
                init_pt = initial_points[c_idx]

            samples, diag = self.sample_chain(
                num_samples=num_samples_per_chain,
                alpha_init=init_pt,
                burn_in=burn_in,
                thinning=thinning,
                seed=chain_seed
            )
            chains.append(samples)
            diagnostics_list.append(diag)

        return chains, diagnostics_list
