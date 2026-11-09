import unittest
import json
from src.app import app


class TestApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_polytope_endpoint(self):
        """Verifies GET /api/polytope returns 3D vertices and simplices."""
        res = self.client.get('/api/polytope')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('vertices', data)
        self.assertIn('faces', data)
        self.assertGreater(len(data['vertices']), 0)
        self.assertGreater(len(data['faces']), 0)

    def test_invalid_rgb_inputs(self):
        """Verifies server rejects invalid or out-of-range RGB inputs with HTTP 400."""
        # Negative value
        res = self.client.post('/api/calculate', json={'r': -0.1, 'g': 0.5, 'b': 0.5})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # Greater than 1.0
        res = self.client.post('/api/calculate', json={'r': 1.5, 'g': 0.5, 'b': 0.5})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # String invalid
        res = self.client.post('/api/calculate', json={'r': 'abc', 'g': 0.5, 'b': 0.5})
        self.assertEqual(res.status_code, 400)

    def test_infeasible_gamut_rgb(self):
        """Verifies that an out-of-gamut target is cleanly rejected without fake data."""
        # Highly saturated green is outside camera gamut
        res = self.client.post(
            '/api/calculate',
            json={'r': 0.05, 'g': 0.95, 'b': 0.05, 'n_samples': 200, 'async': False}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data['feasible'])
        self.assertIn('outside the prior-induced gamut', data.get('message', ''))
        self.assertNotIn('z_points_3d', data)

    def test_valid_rgb_calculation_synchronous(self):
        """Verifies synchronous calculation returns full dimensions, ambiguity, and validation."""
        res = self.client.post(
            '/api/calculate',
            json={
                'r': 0.5, 'g': 0.5, 'b': 0.5,
                'n_samples': 200,
                'burn_in': 20,
                'thinning': 1,
                'bound_mode': '550nm',
                'async': False
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['feasible'])
        self.assertEqual(data['fiber_dimension'], 5)
        self.assertEqual(data['constraint_rank'], 3)
        self.assertIn('dimension_analysis', data)
        self.assertIn('ambiguity_metrics', data)
        self.assertIn('validation', data)
        self.assertIn('spectral_envelope', data)
        self.assertEqual(len(data['spectral_envelope']), 31)

        # Check export endpoint with this data
        for fmt in ['json', 'csv', 'md']:
            exp_res = self.client.post(f'/api/export?format={fmt}', json=data)
            self.assertEqual(exp_res.status_code, 200)
            self.assertGreater(len(exp_res.data), 50)


if __name__ == '__main__':
    unittest.main()
