import unittest
import numpy as np
from src.fiber_geometry import PolytopeGeometryEngine


class TestFiberGeometrySynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PolytopeGeometryEngine(time_budget=30.0)

    def test_canonical_5d_hypercube(self):
        """
        Tests canonical 5D hypercube [-1, 1]^5:
        - Volume: 32.0
        - Surface Area: 160.0
        - Vertices: 32
        - Facets: 10
        - Centroid: [0, 0, 0, 0, 0]
        - Covariance: (1/3) * I_5
        - Inradius: 1.0
        - Centroid Radius: sqrt(5) ~ 2.236068
        - Diameter: 2 * sqrt(5) ~ 4.472136
        """
        # Halfspaces: x_i <= 1, -x_i <= 1  (10 inequalities)
        d = 5
        G = np.vstack([np.eye(d), -np.eye(d)])
        h = np.ones(2 * d)

        res = self.engine.execute_pipeline(G, h)
        self.assertTrue(res["success"])
        self.assertTrue(res["all_integrity_passed"])

        geom = res["geometry"]
        pyr = res["pyramid_moments"]
        ext = res["extents"]
        cheb = res["chebyshev"]

        # Exact Volume = 2^5 = 32
        self.assertAlmostEqual(geom["volume_5d"], 32.0, places=5)
        # Pyramid Volume = 32
        self.assertAlmostEqual(pyr["pyramid_volume"], 32.0, places=5)
        # Surface Area = 10 facets * 16 (4D cube volume) = 160
        self.assertAlmostEqual(geom["surface_area_4d"], 160.0, places=5)
        # Number of vertices = 32
        self.assertEqual(geom["num_vertices"], 32)
        # Number of facets = 10
        self.assertEqual(geom["num_facets"], 10)
        # Inradius = 1.0
        self.assertAlmostEqual(cheb["inradius"], 1.0, places=5)
        # Chebyshev center = 0
        np.testing.assert_allclose(cheb["center"], np.zeros(d), atol=1e-6)
        # Exact Centroid = 0
        np.testing.assert_allclose(pyr["exact_centroid"], np.zeros(d), atol=1e-6)
        # Exact Covariance = (1/3) * I_5
        np.testing.assert_allclose(pyr["exact_covariance"], (1.0 / 3.0) * np.eye(d), atol=1e-5)
        # Centroid Radius = sqrt(5)
        self.assertAlmostEqual(ext["centroid_radius"], np.sqrt(5.0), places=5)
        # Diameter = 2 * sqrt(5)
        self.assertAlmostEqual(ext["diameter"], 2.0 * np.sqrt(5.0), places=5)

    def test_canonical_5d_cross_polytope(self):
        """
        Tests 5D cross-polytope sum |x_i| <= 1:
        - Volume: 2^5 / 5! = 32 / 120 = 4/15 ~ 0.266667
        - Vertices: 10 (±e_i)
        - Facets: 2^5 = 32
        """
        d = 5
        # 32 facets corresponding to all combinations of (+-1, +-1, +-1, +-1, +-1) @ x <= 1
        from itertools import product
        G = np.array(list(product([-1.0, 1.0], repeat=d)))
        h = np.ones(len(G))

        res = self.engine.execute_pipeline(G, h)
        self.assertTrue(res["success"])

        geom = res["geometry"]
        pyr = res["pyramid_moments"]

        # Volume = 4/15
        expected_vol = 4.0 / 15.0
        self.assertAlmostEqual(geom["volume_5d"], expected_vol, places=5)
        self.assertAlmostEqual(pyr["pyramid_volume"], expected_vol, places=5)
        self.assertEqual(geom["num_vertices"], 10)
        self.assertEqual(geom["num_facets"], 32)
        np.testing.assert_allclose(pyr["exact_centroid"], np.zeros(d), atol=1e-6)

    def test_canonical_5d_standard_simplex(self):
        """
        Tests 5D standard simplex: x_i >= 0, sum x_i <= 1
        - Volume: 1 / 5! = 1 / 120 ~ 0.00833333
        - Vertices: 6
        - Facets: 6
        """
        d = 5
        # -x_i <= 0, sum x_i <= 1
        G = np.vstack([-np.eye(d), np.ones((1, d))])
        h = np.zeros(d + 1)
        h[-1] = 1.0

        res = self.engine.execute_pipeline(G, h)
        self.assertTrue(res["success"])

        geom = res["geometry"]
        pyr = res["pyramid_moments"]

        expected_vol = 1.0 / 120.0
        self.assertAlmostEqual(geom["volume_5d"], expected_vol, places=6)
        self.assertEqual(geom["num_vertices"], 6)
        self.assertEqual(geom["num_facets"], 6)
        # Centroid of standard simplex is (1/6) * ones
        np.testing.assert_allclose(pyr["exact_centroid"], np.full(d, 1.0 / 6.0), atol=1e-5)

    def test_anisotropic_box(self):
        """
        Tests anisotropic box with distinct side lengths:
        x_i in [-a_i, a_i], Volume = prod(2*a_i)
        """
        d = 5
        a = np.array([0.5, 1.2, 2.0, 0.8, 1.5])
        G = np.vstack([np.eye(d), -np.eye(d)])
        h = np.hstack([a, a])

        res = self.engine.execute_pipeline(G, h)
        self.assertTrue(res["success"])

        expected_vol = np.prod(2.0 * a)
        self.assertAlmostEqual(res["geometry"]["volume_5d"], expected_vol, places=4)
        # Covariance diagonal = (1/3) * a_i^2
        expected_cov_diag = (1.0 / 3.0) * (a ** 2)
        np.testing.assert_allclose(np.diag(res["pyramid_moments"]["exact_covariance"]), expected_cov_diag, atol=1e-4)

    def test_translated_and_rotated_invariance(self):
        """
        Tests that translation by 1e3 preserves exact volume and covariance.
        """
        d = 5
        # Hypercube [-1, 1]^5 translated by 1000 * ones
        shift = 1000.0 * np.ones(d)
        G = np.vstack([np.eye(d), -np.eye(d)])
        h = np.hstack([1.0 + shift, 1.0 - shift])

        res = self.engine.execute_pipeline(G, h)
        self.assertTrue(res["success"])
        self.assertAlmostEqual(res["geometry"]["volume_5d"], 32.0, places=4)
        np.testing.assert_allclose(res["pyramid_moments"]["exact_centroid"], shift, atol=1e-4)
        np.testing.assert_allclose(res["pyramid_moments"]["exact_covariance"], (1.0 / 3.0) * np.eye(d), atol=1e-4)

    def test_row_shuffle_and_redundancy_invariance(self):
        """
        Tests that row shuffling, duplicate rows, and zero-norm rows do not affect geometry.
        """
        d = 5
        G = np.vstack([np.eye(d), -np.eye(d)])
        h = np.ones(2 * d)

        # Add duplicates with looser h and zero-norm rows
        rng = np.random.RandomState(42)
        G_dup = np.vstack([G, G * 2.0, np.zeros((3, d))])
        h_dup = np.hstack([h, h * 2.5, np.array([1.0, 5.0, 0.0])]) # looser bounds and 0 <= pos

        # Shuffle rows
        p = rng.permutation(len(G_dup))
        G_shuffled = G_dup[p]
        h_shuffled = h_dup[p]

        res = self.engine.execute_pipeline(G_shuffled, h_shuffled)
        self.assertTrue(res["success"])
        self.assertAlmostEqual(res["geometry"]["volume_5d"], 32.0, places=4)
        self.assertEqual(res["geometry"]["num_vertices"], 32)


if __name__ == '__main__':
    unittest.main()
