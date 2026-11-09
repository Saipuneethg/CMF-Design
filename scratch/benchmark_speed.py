import os
import time
import numpy as np
import colour
from scipy.optimize import minimize

print("Testing loading of 13 curves and parameter recovery...")
base_dir = os.path.abspath(".")
artifacts_dir = os.path.join(base_dir, "artifacts")
curves_dir = os.path.join(base_dir, "camera comparisions", "reports", "Experiment_4_Initialization_Independence_20261031_221323", "Final_Optimized_Curves")

wavelengths = np.load(os.path.join(artifacts_dir, "wavelengths.npy")).astype(np.float64)
pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)
A_red = np.load(os.path.join(artifacts_dir, "A_red.npy")).astype(np.float64)
b_red = np.load(os.path.join(artifacts_dir, "b_red.npy")).astype(np.float64)
M_baseline = np.load(os.path.join(artifacts_dir, "M_camera.npy")).astype(np.float64)
target_throughput = float(np.mean(np.sum(M_baseline, axis=1)))

sub_idx = np.linspace(0, len(A_red) - 1, 40000, dtype=int)
A_sub = A_red[sub_idx]
b_sub = b_red[sub_idx]

z_train_path = os.path.join(base_dir, "results", "global_cmf_reoptimization_20261028_205403", "data", "Z_train.npy")
z_test_path = os.path.join(base_dir, "results", "global_cmf_reoptimization_20261028_205403", "data", "Z_test.npy")
r_test_path = os.path.join(base_dir, "results", "global_cmf_reoptimization_20261028_205403", "data", "R_test.npy")
Z_train = np.load(z_train_path).astype(np.float64)
Z_test = np.load(z_test_path).astype(np.float64)
R_test = np.load(r_test_path).astype(np.float64)
R_train = np.clip(pca_mean + Z_train @ pca_basis.T, 0.0, 1.0)

H_train = np.maximum(b_sub[:, None] - A_sub @ Z_train.T, 1e-12)
H_test = np.maximum(b_sub[:, None] - A_sub @ Z_test.T, 1e-12)

# CIE & D65
cmf = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
cie_xyz_cmf = cmf[wavelengths].T.astype(np.float64)
illum_D65 = colour.SDS_ILLUMINANTS['D65']
d65_raw = illum_D65[wavelengths].astype(np.float64)
k_d65 = 100.0 / np.sum(d65_raw * cie_xyz_cmf[1])
d65_scaled = d65_raw * k_d65
xy_w = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
XYZ_train_ref = (R_train * d65_scaled) @ cie_xyz_cmf.T
XYZ_test_ref = (R_test * d65_scaled) @ cie_xyz_cmf.T

def params_to_matrix(params):
    wls = wavelengths
    a_b, mu_b, sig_b = abs(params[0]), params[1], max(abs(params[2]), 5.0)
    a_g, mu_g, sig_g = abs(params[3]), params[4], max(abs(params[5]), 5.0)
    a_r1, mu_r1, sig_r1 = abs(params[6]), params[7], max(abs(params[8]), 5.0)
    a_r2, mu_r2, sig_r2 = abs(params[9]), params[10], max(abs(params[11]), 5.0)

    s_b = a_b * np.exp(-0.5 * ((wls - mu_b) / sig_b) ** 2)
    s_g = a_g * np.exp(-0.5 * ((wls - mu_g) / sig_g) ** 2)
    s_r = (a_r1 * np.exp(-0.5 * ((wls - mu_r1) / sig_r1) ** 2) +
           a_r2 * np.exp(-0.5 * ((wls - mu_r2) / sig_r2) ** 2))

    sum_b = np.sum(s_b)
    sum_g = np.sum(s_g)
    sum_r = np.sum(s_r)

    s_b = (s_b / sum_b) * target_throughput if sum_b > 1e-9 else np.ones(31) * (target_throughput / 31)
    s_g = (s_g / sum_g) * target_throughput if sum_g > 1e-9 else np.ones(31) * (target_throughput / 31)
    s_r = (s_r / sum_r) * target_throughput if sum_r > 1e-9 else np.ones(31) * (target_throughput / 31)
    return np.array([s_r, s_g, s_b], dtype=np.float64)

bounds = [
    (0.2, 5.0), (420.0, 475.0), (15.0, 60.0),
    (0.2, 5.0), (515.0, 575.0), (20.0, 65.0),
    (0.2, 5.0), (590.0, 660.0), (20.0, 75.0),
    (0.0, 1.5), (420.0, 470.0), (10.0, 45.0)
]

def fit_matrix_to_params(M_target):
    M_norm = np.zeros_like(M_target)
    for c in range(3):
        s = np.sum(M_target[c])
        M_norm[c] = (M_target[c] / s) * target_throughput if s > 1e-9 else M_target[c]
    centroids = [np.sum(wavelengths * M_norm[c]) / np.sum(M_norm[c]) for c in range(3)]
    order = np.argsort(centroids)[::-1]
    M_ordered = M_norm[order]
    b_peak = wavelengths[np.argmax(M_ordered[2])]
    g_peak = wavelengths[np.argmax(M_ordered[1])]
    r_peak = wavelengths[np.argmax(M_ordered[0])]
    p0 = [
        2.0, np.clip(b_peak, 430.0, 470.0), 25.0,
        2.0, np.clip(g_peak, 520.0, 560.0), 35.0,
        2.0, np.clip(r_peak, 600.0, 640.0), 40.0,
        0.2, 445.0, 20.0
    ]
    def loss(p):
        M_fit = params_to_matrix(p)
        return np.sum((M_fit - M_ordered) ** 2)
    res = minimize(loss, p0, bounds=bounds, method='L-BFGS-B')
    return res.x, params_to_matrix(res.x)

def evaluate_ambiguity_fast(M, H):
    A_theta = M @ pca_basis
    U, S, Vt = np.linalg.svd(A_theta)
    if S[-1] < 1e-4:
        return 999.0, np.ones(H.shape[1]) * 999.0
    N_theta = Vt[3:].T
    G = A_sub @ N_theta
    n_pts = H.shape[1]
    spans = np.zeros((n_pts, 5))
    for j in range(5):
        G_j = G[:, j]
        p_mask = G_j > 1e-9
        t_p = np.min(H[p_mask] / G_j[p_mask, None], axis=0)
        n_mask = G_j < -1e-9
        t_n = np.min(H[n_mask] / (-G_j[n_mask, None]), axis=0)
        spans[:, j] = t_p + t_n
    spreads = np.sqrt(np.sum(spans ** 2, axis=1) / 12.0)
    return float(np.mean(spreads)), spreads

# Test fitting 13 curves
camera_names = [
    "Canon_1DMarkIII", "Nikon_D200", "Point_Grey_Grasshopper",
    "CIE_1931_Derived", "Original_Baseline_Camera",
    "Random_Smooth_1_seed101", "Random_Smooth_2_seed202",
    "Random_Smooth_3_seed303", "Random_Smooth_4_seed404",
    "Random_Smooth_5_seed505", "Random_Smooth_6_seed606",
    "Random_Smooth_7_seed707", "Random_Smooth_8_seed808"
]

recovered_params = {}
fitted_matrices = {}
original_matrices = {}

for name in camera_names:
    npy_file = os.path.join(curves_dir, f"{name}_optimized_sensitivity.npy")
    M_orig = np.load(npy_file)
    p_fit, M_fit = fit_matrix_to_params(M_orig)
    err = np.max(np.abs(M_fit - M_orig))
    recovered_params[name] = p_fit
    fitted_matrices[name] = M_fit
    original_matrices[name] = M_orig
    print(f"{name:30s} max abs fit error: {err:.6e}")

# Speed benchmark of evaluate_ambiguity_fast
t0 = time.time()
n_test_evals = 100
for _ in range(n_test_evals):
    mean_s, _ = evaluate_ambiguity_fast(fitted_matrices["Canon_1DMarkIII"], H_test)
t1 = time.time()
print(f"\n100 evaluations took {t1 - t0:.3f} s ({((t1 - t0)/n_test_evals)*1000:.2f} ms per eval)")
print(f"Projected time for 7,878 evaluations: {((t1 - t0)/n_test_evals) * 7878:.2f} seconds!")
