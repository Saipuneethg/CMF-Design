"""
test_cmf_image_generation.py
============================
Unit & Regression Tests for Strict One-CMF-per-Image Rule.

Verifies:
1. Rejection of multi-CMF inputs, concatenated matrices, or invalid dimensions.
2. Buffer isolation: rendering CMF A, then CMF B, then CMF A produces byte-for-byte identical output.
3. Independent rendering: each CMF produces its own output image with isolated metadata.
4. No cross-CMF blending, averaging, or accumulation.
5. Correct output dimensions (H, W, 3), RGB channel ordering, and uint8 encoding.
6. Source spectral data and original RGB image remain byte-for-byte unmodified.
7. Complete discovery of Global Optimized CMF, Baseline, CIE Observers, and 28 Physical Cameras.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.cmf_image_renderer import CMFRenderer, load_all_available_cmfs, compute_file_sha256


class TestCMFImageGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.spec_path = os.path.join(cls.root_dir, "Spectral", "000002.npy")
        cls.rgb_path = os.path.join(cls.root_dir, "Dataset", "RGB", "000002.png")
        cls.cmfs = load_all_available_cmfs(cls.root_dir)

        # Baseline and Global Optimized CMFs
        cls.global_opt = next(c for c in cls.cmfs if c["id"] == "global_optimized_cmf")
        cls.baseline = next(c for c in cls.cmfs if c["id"] == "original_calibrated_camera")
        cls.cie_1931 = next(c for c in cls.cmfs if c["id"] == "cie_1931_2deg")

        # Load small test crop (32x32x31) from real spectral cube for rapid testing
        if os.path.exists(cls.spec_path):
            full_cube = np.load(cls.spec_path)
            cls.test_spec = full_cube[:32, :32, :]
        else:
            cls.test_spec = np.random.uniform(500, 30000, (32, 32, 31)).astype(np.float32)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cmf_render_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_all_cmf_discovery(self):
        """Verify that all 32 CMFs are discovered with valid 3x31 shape and non-negative curves."""
        self.assertGreaterEqual(len(self.cmfs), 30, "Expected at least 30 CMFs (2 models + 2 CIE + 28 cameras)")

        ids = [c["id"] for c in self.cmfs]
        self.assertIn("global_optimized_cmf", ids)
        self.assertIn("original_calibrated_camera", ids)
        self.assertIn("cie_1931_2deg", ids)
        self.assertIn("cie_1964_10deg", ids)

        for c in self.cmfs:
            M = c["matrix"]
            self.assertEqual(M.shape, (3, 31), f"CMF {c['name']} must have shape (3, 31)")
            self.assertGreaterEqual(M.min(), -0.1, f"CMF {c['name']} has severe negative sensitivities: min={M.min()}")
            self.assertFalse(np.isnan(M).any(), f"CMF {c['name']} has NaN")

    def test_strict_single_cmf_enforcement(self):
        """Verify that CMFRenderer strictly rejects multiple concatenated CMFs or invalid shapes."""
        # Case 1: Concatenated two CMFs (shape 6 x 31) -> MUST FAIL
        two_cmfs = np.vstack([self.global_opt["matrix"], self.baseline["matrix"]])
        invalid_def_6ch = {"id": "two_cmfs", "name": "Illegal Two CMFs", "matrix": two_cmfs}
        with self.assertRaises(ValueError) as ctx:
            CMFRenderer.validate_single_cmf(invalid_def_6ch)
        self.assertIn("STRICT ONE-CMF RULE VIOLATION", str(ctx.exception))

        # Case 2: 1 channel only -> MUST FAIL
        invalid_def_1ch = {"id": "one_ch", "name": "1 Channel", "matrix": self.global_opt["matrix"][:1, :]}
        with self.assertRaises(ValueError):
            CMFRenderer.validate_single_cmf(invalid_def_1ch)

        # Case 3: Wrong number of bands (e.g. 3 x 25) -> MUST FAIL
        invalid_def_bands = {"id": "wrong_bands", "name": "Wrong Bands", "matrix": self.global_opt["matrix"][:, :25]}
        with self.assertRaises(ValueError):
            CMFRenderer.validate_single_cmf(invalid_def_bands)

    def test_buffer_isolation_and_no_accumulation(self):
        """Verify rendering A, then B, then A produces identical results with no state leakage."""
        out_a1 = os.path.join(self.temp_dir, "render_a1.png")
        out_b = os.path.join(self.temp_dir, "render_b.png")
        out_a2 = os.path.join(self.temp_dir, "render_a2.png")

        img_a1, meta_a1 = CMFRenderer.render_single_cmf_image(self.test_spec, self.global_opt, out_a1)
        img_b, meta_b = CMFRenderer.render_single_cmf_image(self.test_spec, self.baseline, out_b)
        img_a2, meta_a2 = CMFRenderer.render_single_cmf_image(self.test_spec, self.global_opt, out_a2)

        # Verify A1 and A2 are 100% identical byte-for-byte
        np.testing.assert_array_equal(img_a1, img_a2, "Buffer pollution: rendering B altered subsequent rendering of A")
        self.assertEqual(meta_a1["saved_file_sha256"], meta_a2["saved_file_sha256"])

        # Verify A and B are rendered independently
        self.assertEqual(meta_a1["cmf_id"], "global_optimized_cmf")
        self.assertEqual(meta_b["cmf_id"], "original_calibrated_camera")

    def test_independent_renderings_metadata_and_hashes(self):
        """Verify that each rendered CMF produces independent files with verified single-CMF metadata."""
        selected_cmfs = [self.global_opt, self.baseline, self.cie_1931]

        for cmf in selected_cmfs:
            out_p = os.path.join(self.temp_dir, f"{cmf['slug']}.png")
            img, meta = CMFRenderer.render_single_cmf_image(self.test_spec, cmf, out_p)

            # File exists on disk
            self.assertTrue(os.path.exists(out_p))
            self.assertGreater(os.path.getsize(out_p), 100)

            # Metadata verification
            self.assertTrue(meta["strict_single_cmf_verified"])
            self.assertEqual(meta["cmf_id"], cmf["id"])
            self.assertEqual(meta["cmf_channels_count"], 3)
            self.assertEqual(meta["output_dimensions"], [32, 32, 3])
            self.assertEqual(meta["output_channels"], ["R", "G", "B"])

            # Image array verification
            self.assertEqual(img.shape, (32, 32, 3))
            self.assertEqual(img.dtype, np.uint8)
            self.assertGreaterEqual(int(img.min()), 0)
            self.assertLessEqual(int(img.max()), 255)

    def test_no_cross_cmf_blending(self):
        """Verify that output RGB is strictly derived from single CMF and not a blended ensemble."""
        img_opt, _ = CMFRenderer.render_single_cmf_image(self.test_spec, self.global_opt)
        img_base, _ = CMFRenderer.render_single_cmf_image(self.test_spec, self.baseline)

        # Theoretical 50/50 blend
        blended = (0.5 * img_opt.astype(np.float64) + 0.5 * img_base.astype(np.float64)).astype(np.uint8)

        # Real single-CMF rendering must NOT equal an artificial blend
        diff_from_blend_opt = np.abs(img_opt.astype(np.int32) - blended.astype(np.int32)).mean()
        self.assertGreater(diff_from_blend_opt, 0.0, "Rendered image appears to be an unintended blend")

    def test_source_data_immutability(self):
        """Verify that rendering multiple CMFs leaves source spectral data and RGB original 100% unchanged."""
        if not (os.path.exists(self.spec_path) and os.path.exists(self.rgb_path)):
            self.skipTest("Source test files not found")

        h_spec_before = compute_file_sha256(self.spec_path)
        h_rgb_before = compute_file_sha256(self.rgb_path)

        # Perform 3 independent rendering operations
        full_spec = np.load(self.spec_path)
        CMFRenderer.render_single_cmf_image(full_spec, self.global_opt, os.path.join(self.temp_dir, "t1.png"))
        CMFRenderer.render_single_cmf_image(full_spec, self.baseline, os.path.join(self.temp_dir, "t2.png"))
        CMFRenderer.render_single_cmf_image(full_spec, self.cie_1931, os.path.join(self.temp_dir, "t3.png"))

        h_spec_after = compute_file_sha256(self.spec_path)
        h_rgb_after = compute_file_sha256(self.rgb_path)

        self.assertEqual(h_spec_before, h_spec_after, "Source spectral cube was modified during rendering!")
        self.assertEqual(h_rgb_before, h_rgb_after, "Original RGB image was modified during rendering!")

    def test_api_cmf_list_and_render_endpoint(self):
        """Verify Flask API endpoints /api/cmf/list and /api/cmf/render_image."""
        from src.app import app
        client = app.test_client()

        # 1. Test /api/cmf/list
        res = client.get('/api/cmf/list')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreaterEqual(data["count"], 30)
        self.assertTrue(any(c["id"] == "global_optimized_cmf" for c in data["cmfs"]))

        # 2. Test /api/cmf/render_image for Global Optimized CMF
        res_render = client.get('/api/cmf/render_image?cmf_id=global_optimized_cmf&image_id=000002')
        self.assertEqual(res_render.status_code, 200)
        data_render = res_render.get_json()
        self.assertTrue(data_render["success"])
        self.assertEqual(data_render["cmf_id"], "global_optimized_cmf")
        self.assertTrue(data_render["image_data"].startswith("data:image/png;base64,"))
        self.assertTrue(data_render["metadata"]["strict_single_cmf_verified"])

        # 3. Test non-existent CMF returns 404
        res_bad = client.get('/api/cmf/render_image?cmf_id=nonexistent_cmf_123&image_id=000002')
        self.assertEqual(res_bad.status_code, 404)


if __name__ == "__main__":
    unittest.main()

