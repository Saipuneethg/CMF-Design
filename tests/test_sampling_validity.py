import unittest
import numpy as np
import os
from src.metamer_solver import MetamerSolver
from src.hit_and_run import HitAndRunSampler
from src.validation import Validator


class TestSamplingValidity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cls.solver = MetamerSolver(cls.artifacts_dir)
        cls.validator = Validator(artifacts_dir=cls.artifacts_dir)
        cls.rgb = [0.5, 0.5, 0.5]
        cls.fiber = cls.solver.construct_fiber(cls.rgb)

    def test_samples_satisfy_polytope(self):
        """Verifies that all Hit-and-Run points strictly satisfy G @ alpha <= h."""
        G = self.fiber["G"]
        h = self.fiber["h"]
        sampler = HitAndRunSampler(G, h)
        
        samples, diag = sampler.sample_chain(num_samples=100, burn_in=20, thinning=2, seed=42)
        self.assertEqual(len(samples), 100)
        self.assertLessEqual(diag["max_violation_retained"], 1e-5)
        self.assertEqual(diag["valid_samples_retained"], 100)

    def test_seed_reproducibility(self):
        """Verifies that running two chains with the exact same seed yields identical samples."""
        G = self.fiber["G"]
        h = self.fiber["h"]
        sampler = HitAndRunSampler(G, h)

        s1, _ = sampler.sample_chain(num_samples=50, seed=1234, burn_in=10, thinning=1)
        s2, _ = sampler.sample_chain(num_samples=50, seed=1234, burn_in=10, thinning=1)

        np.testing.assert_array_equal(s1, s2, "Chains with identical seed must produce identical samples")

    def test_multi_chain_agreement(self):
        """Verifies that independent chains stay within reasonable distance and have low subspace distance."""
        G = self.fiber["G"]
        h = self.fiber["h"]
        N = self.fiber["N"]
        z0 = self.fiber["z0"]
        sampler = HitAndRunSampler(G, h)

        chains_alpha, diags = sampler.sample_multiple_chains(
            num_chains=2, num_samples_per_chain=150, burn_in=20, thinning=2, base_seed=100
        )
        chains_z = [z0 + ch @ N.T for ch in chains_alpha]

        comp = Validator.compare_chains(chains_z)
        self.assertIn("mean_centroid_distance", comp)
        self.assertLess(comp["mean_centroid_distance"], 1.0)
        self.assertLess(comp["mean_subspace_distance"], 1.0)

    def test_reconstructed_rgb_fidelity(self):
        """Verifies that spectral samples reconstructed from z satisfy ||M R - c|| < 1e-5."""
        res = self.solver.solve(self.rgb, num_points=100, seed=42)
        val = self.validator.validate_points(self.rgb, res["z_points"])
        
        self.assertLess(val["mean_error"], 1e-5, f"Mean RGB error too large: {val['mean_error']}")
        self.assertLess(val["max_error"], 1e-5, f"Max RGB error too large: {val['max_error']}")
        self.assertLess(val["rms_error"], 1e-5, f"RMS RGB error too large: {val['rms_error']}")
        self.assertEqual(val["valid_points"], len(res["z_points"]))


if __name__ == '__main__':
    unittest.main()
