import unittest
import numpy as np
import os
import time
from src.app import app
from src.metamer_solver import MetamerSolver


class TestUserSampleCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cls.solver = MetamerSolver(cls.artifacts_dir)
        cls.feasible_rgb = [0.5, 0.5, 0.5]
        cls.outside_rgb = [0.05, 0.95, 0.05]

    def test_valid_n_returns_exact_samples_solver(self):
        """Verifies that solver returns exactly N samples for different values of N."""
        for N in [100, 250]:
            res = self.solver.solve(self.feasible_rgb, num_points=N, seed=42)
            self.assertTrue(res["feasible"])
            self.assertEqual(len(res["alpha_points"]), N)
            self.assertEqual(len(res["z_points"]), N)
            self.assertEqual(res["n_samples"], N)

    def test_valid_n_returns_exact_samples_api(self):
        """Verifies that the /api/calculate endpoint returns exactly N 3D points."""
        res = self.client.post('/api/calculate', json={
            'r': 0.5, 'g': 0.5, 'b': 0.5,
            'n_samples': 150
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['feasible'])
        self.assertEqual(len(data['z_points_3d']), 150)
        self.assertEqual(data['n_samples'], 150)
        self.assertEqual(data['validation']['total_points'], 150)

    def test_n_below_min_returns_400(self):
        """Verifies that N < 100 returns HTTP 400 with an informative error message."""
        for bad_n in [99, 50, 0, -10]:
            res = self.client.post('/api/calculate', json={
                'r': 0.5, 'g': 0.5, 'b': 0.5,
                'n_samples': bad_n
            })
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn('error', data)
            self.assertIn('between 100 and 5000', data['error'])

    def test_n_above_max_returns_400(self):
        """Verifies that N > 5000 returns HTTP 400 with an informative error message."""
        for bad_n in [5001, 6000, 10000]:
            res = self.client.post('/api/calculate', json={
                'r': 0.5, 'g': 0.5, 'b': 0.5,
                'n_samples': bad_n
            })
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn('error', data)
            self.assertIn('between 100 and 5000', data['error'])

    def test_non_integer_n_returns_400(self):
        """Verifies that non-integer N (floats, strings, booleans) returns HTTP 400."""
        invalid_inputs = [250.5, "one_thousand", True, None, [1000]]
        for bad_val in invalid_inputs:
            res = self.client.post('/api/calculate', json={
                'r': 0.5, 'g': 0.5, 'b': 0.5,
                'n_samples': bad_val
            })
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn('error', data)

    def test_invalid_burn_in_and_thinning_return_400(self):
        """Verifies that burn_in < 0 or thinning < 1 returns HTTP 400."""
        # Negative burn-in
        res1 = self.client.post('/api/calculate', json={
            'r': 0.5, 'g': 0.5, 'b': 0.5,
            'n_samples': 100,
            'burn_in': -5
        })
        self.assertEqual(res1.status_code, 400)

        # Thinning < 1
        res2 = self.client.post('/api/calculate', json={
            'r': 0.5, 'g': 0.5, 'b': 0.5,
            'n_samples': 100,
            'thinning': 0
        })
        self.assertEqual(res2.status_code, 400)

    def test_outside_gamut_target_rejection_for_any_n(self):
        """Verifies that outside-gamut target is cleanly rejected regardless of N."""
        for N in [100, 500, 1000]:
            res = self.client.post('/api/calculate', json={
                'r': self.outside_rgb[0],
                'g': self.outside_rgb[1],
                'b': self.outside_rgb[2],
                'n_samples': N
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertFalse(data['feasible'])
            self.assertNotIn('z_points_3d', data)

    def test_exact_lp_bounds_identical_across_n(self):
        """
        Verifies that exact LP bounds are mathematically identical for N=100 and N=1000,
        confirming that physical bounds are independent of sample count.
        """
        res_100 = self.solver.solve(
            self.feasible_rgb,
            num_points=100,
            seed=42,
            compute_bounds=True,
            bound_wavelengths=[15] # 550nm
        )
        res_1000 = self.solver.solve(
            self.feasible_rgb,
            num_points=1000,
            seed=42,
            compute_bounds=True,
            bound_wavelengths=[15] # 550nm
        )

        b_100 = res_100["exact_bounds"][0]
        b_1000 = res_1000["exact_bounds"][0]

        self.assertEqual(b_100["wavelength_nm"], 550.0)
        self.assertAlmostEqual(b_100["exact_min"], b_1000["exact_min"], places=10)
        self.assertAlmostEqual(b_100["exact_max"], b_1000["exact_max"], places=10)
        self.assertAlmostEqual(b_100["exact_range"], b_1000["exact_range"], places=10)

    def test_async_job_polling_flow(self):
        """Verifies that asynchronous calculation with background polling succeeds."""
        # 1. Start job
        init_res = self.client.post('/api/calculate', json={
            'r': 0.5, 'g': 0.5, 'b': 0.5,
            'n_samples': 100,
            'async': True
        })
        self.assertEqual(init_res.status_code, 202)
        init_data = init_res.get_json()
        self.assertIn('job_id', init_data)
        job_id = init_data['job_id']

        # 2. Poll until completed (timeout after 60s)
        completed = False
        final_job = None
        for _ in range(120):
            time.sleep(0.5)
            poll_res = self.client.get(f'/api/jobs/{job_id}')
            self.assertEqual(poll_res.status_code, 200)
            final_job = poll_res.get_json()
            if final_job['status'] == 'completed':
                completed = True
                break

        self.assertTrue(completed, "Async job failed to complete within timeout")
        self.assertIsNotNone(final_job['result'])
        self.assertEqual(len(final_job['result']['z_points_3d']), 100)


if __name__ == '__main__':
    unittest.main()
