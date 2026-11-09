import unittest
import numpy as np
from src.ambiguity_metrics import AmbiguityMetrics


class TestSyntheticGeometry(unittest.TestCase):
    def test_canonical_hypercube_geometry(self):
        """
        Validates sample statistics and volume scale estimates against
        analytically known ground-truth for a 5-dimensional hypercube [-1, 1]^5.
        """
        dim = 5
        num_samples = 10000
        val = AmbiguityMetrics.validate_synthetic_geometry(dim=dim, num_samples=num_samples, seed=42)

        # 1. Coordinate mean must be near 0
        self.assertLess(val["sample_mean_norm"], 0.05, "Sample mean deviates too far from 0")

        # 2. Variance per coordinate must be close to 1/3 ~ 0.3333
        self.assertLess(
            val["variance_relative_error"],
            0.05,
            f"Coordinate variance relative error {val['variance_relative_error']:.4f} exceeds 5%"
        )

        # 3. Volume scale estimate (det(Cov))^(1/(2d)) must be close to 1/sqrt(3) ~ 0.57735
        self.assertLess(
            val["volume_scale_relative_error"],
            0.05,
            f"Volume scale relative error {val['volume_scale_relative_error']:.4f} exceeds 5%"
        )

        # 4. Box volume must be close to 2^5 = 32
        self.assertLess(
            val["box_volume_relative_error"],
            0.15,
            f"Box volume relative error {val['box_volume_relative_error']:.4f} exceeds 15%"
        )

    def test_fiber_dimension_formula(self):
        """Verifies mathematical dimension computation on synthetic matrices."""
        # 3 x 8 rank-3 matrix
        A_syn = np.random.RandomState(42).randn(3, 8)
        dim_info = AmbiguityMetrics.compute_fiber_dimension(A_syn, K=8)
        self.assertEqual(dim_info["camera_rank"], 3)
        self.assertEqual(dim_info["fiber_dimension"], 5)

        # 3 x 8 rank-2 matrix
        A_rank2 = np.copy(A_syn)
        A_rank2[2] = A_rank2[0] + A_rank2[1]
        dim_info2 = AmbiguityMetrics.compute_fiber_dimension(A_rank2, K=8)
        self.assertEqual(dim_info2["camera_rank"], 2)
        self.assertEqual(dim_info2["fiber_dimension"], 6)


if __name__ == '__main__':
    unittest.main()
