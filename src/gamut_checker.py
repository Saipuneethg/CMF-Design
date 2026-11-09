import numpy as np
from scipy.optimize import linprog


class GamutChecker:
    """
    Evaluates whether an RGB observation c falls within the learned spectral
    polytope gamut, and computes feasible anchor points z0 in PCA space satisfying:
    
        A_camera @ z0 = c - camera_offset
        A_red @ z0 <= b_red
    """
    def __init__(self, A_red, b_red, A_camera, camera_offset, tol=1e-5):
        self.A_red = np.asarray(A_red, dtype=np.float64)
        self.b_red = np.asarray(b_red, dtype=np.float64)
        self.A_camera = np.asarray(A_camera, dtype=np.float64)
        self.camera_offset = np.asarray(camera_offset, dtype=np.float64)
        self.tol = tol
        self.dim = self.A_camera.shape[1]

    def check_feasibility(self, rgb):
        """
        Preserves backward compatibility: returns (bool feasible, np.ndarray or None z0).
        """
        c = np.array(rgb, dtype=np.float64).flatten()
        b_eq = c - self.camera_offset
        c_obj = np.zeros(self.dim)
        bounds = [(None, None) for _ in range(self.dim)]

        res = linprog(
            c_obj,
            A_ub=self.A_red,
            b_ub=self.b_red,
            A_eq=self.A_camera,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if not res.success or res.x is None:
            return False, None

        z0 = res.x
        # Verify tolerances explicitly
        eq_res = np.linalg.norm(self.A_camera @ z0 - b_eq)
        ineq_viol = np.max(self.A_red @ z0 - self.b_red)
        if eq_res > self.tol or ineq_viol > self.tol:
            # If solver claimed success but violated tolerances, reject
            return False, None

        return True, z0

    def find_multiple_feasible_points(self, rgb, num_points=3, seed=42):
        """
        Attempts to find multiple distinct feasible anchor points in PCA space
        by optimizing in different random directions.
        
        Returns:
        - points: List of distinct feasible vectors z0 in R^K.
        """
        feasible, z0 = self.check_feasibility(rgb)
        if not feasible:
            return []

        points = [z0]
        if num_points <= 1:
            return points

        rng = np.random.RandomState(seed)
        c = np.array(rgb, dtype=np.float64).flatten()
        b_eq = c - self.camera_offset
        bounds = [(None, None) for _ in range(self.dim)]

        for attempt in range((num_points - 1) * 2):
            direction = rng.randn(self.dim)
            res = linprog(
                direction,
                A_ub=self.A_red,
                b_ub=self.b_red,
                A_eq=self.A_camera,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            if res.success and res.x is not None:
                cand = res.x
                # Check feasibility and distinctness
                eq_res = np.linalg.norm(self.A_camera @ cand - b_eq)
                ineq_viol = np.max(self.A_red @ cand - self.b_red)
                if eq_res <= self.tol and ineq_viol <= self.tol:
                    # Check distance from existing points
                    is_distinct = all(np.linalg.norm(cand - p) > 1e-3 for p in points)
                    if is_distinct:
                        points.append(cand)
                        if len(points) == num_points:
                            break

        return points
