import unittest
import numpy as np
import os
from src.metamer_solver import MetamerSolver
from src.gamut_checker import GamutChecker


class TestFiberConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cls.solver = MetamerSolver(cls.artifacts_dir)

    def test_null_space_orthogonality(self):
        """Verifies that A_camera @ N == 0 exactly."""
        A = self.solver.A_camera
        N = self.solver.N
        res = A @ N
        max_err = np.max(np.abs(res))
        self.assertLess(max_err, 1e-12, f"Null space condition violated: max error = {max_err}")

    def test_fiber_dimension_calculation(self):
        """Verifies that fiber dimension is dynamically computed as K - rank(A)."""
        K = self.solver.K
        rank = self.solver.constraint_rank
        d_f = self.solver.fiber_dimension
        self.assertEqual(rank, 3, f"Expected rank 3, got {rank}")
        self.assertEqual(d_f, K - rank, "Fiber dimension must equal K - rank(A)")
        self.assertEqual(d_f, 5, f"Expected fiber dimension 5 for K=8, rank=3, got {d_f}")
        self.assertEqual(self.solver.N.shape[1], d_f, "N column count must match fiber dimension")

    def test_feasible_anchor_constraints(self):
        """Verifies that anchor z0 satisfies RGB equality and polytope inequalities within tolerance."""
        rgb = [0.5, 0.5, 0.5]
        fiber = self.solver.construct_fiber(rgb)
        self.assertTrue(fiber["feasible"], "RGB [0.5, 0.5, 0.5] must be feasible")

        z0 = fiber["z0"]
        c = np.array(rgb)
        b_eq = c - self.solver.camera_offset

        # Equality residual: ||A z0 - b||
        eq_res = np.linalg.norm(self.solver.A_camera @ z0 - b_eq)
        self.assertLess(eq_res, 1e-5, f"Equality constraint residual too large: {eq_res}")

        # Inequality violation: max(A_red @ z0 - b_red)
        ineq_viol = np.max(self.solver.A_red @ z0 - self.solver.b_red)
        self.assertLess(ineq_viol, 1e-5, f"Polytope inequality violated: {ineq_viol}")

    def test_reduced_inequality_definition(self):
        """Verifies G = C @ N and h = d - C @ z0."""
        rgb = [0.5, 0.5, 0.5]
        fiber = self.solver.construct_fiber(rgb)
        z0 = fiber["z0"]
        G = fiber["G"]
        h = fiber["h"]

        expected_G = self.solver.A_red @ self.solver.N
        expected_h = self.solver.b_red - self.solver.A_red @ z0

        np.testing.assert_allclose(G, expected_G, atol=1e-12)
        np.testing.assert_allclose(h, expected_h, atol=1e-12)

    def test_out_of_gamut_rejection(self):
        """Verifies that an impossible color observation is correctly rejected."""
        rgb_impossible = [0.05, 0.95, 0.05] # saturated green outside natural prior
        fiber = self.solver.construct_fiber(rgb_impossible)
        self.assertFalse(fiber["feasible"], "Out-of-gamut RGB should not yield feasible fiber")


if __name__ == '__main__':
    unittest.main()
