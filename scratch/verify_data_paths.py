import os, sys
import json
import numpy as np

# Verify all data paths
artifacts_dir = "artifacts"
assert os.path.exists(os.path.join(artifacts_dir, "wavelengths.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "pca_mean.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "pca_basis.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "A_red.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "b_red.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "M_camera.npy"))
assert os.path.exists(os.path.join(artifacts_dir, "M_camera_optimized.npy"))

cam_inv = "camera comparisions/metadata/camera_database_inventory.json"
cam_mats = "camera comparisions/sensitivity_functions/camera_sensitivity_matrices.npy"
assert os.path.exists(cam_inv)
assert os.path.exists(cam_mats)

z_train_path = "results/global_cmf_reoptimization_20261028_205403/data/Z_train.npy"
z_test_path = "results/global_cmf_reoptimization_20261028_205403/data/Z_test.npy"
assert os.path.exists(z_train_path)
assert os.path.exists(z_test_path)

print("All required data paths verified successfully!")
