import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import colour

# Load artifacts
artifacts_dir = "artifacts"
wavelengths = np.linspace(400, 700, 31)
pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)
A_red = np.load(os.path.join(artifacts_dir, "A_red.npy")).astype(np.float64)
b_red = np.load(os.path.join(artifacts_dir, "b_red.npy")).astype(np.float64)

sub_idx = np.linspace(0, len(A_red) - 1, 40000, dtype=int)
A_sub = A_red[sub_idx]
b_sub = b_red[sub_idx]

M_baseline = np.load(os.path.join(artifacts_dir, "M_camera.npy")).astype(np.float64)
target_throughput = float(np.mean(np.sum(M_baseline, axis=1)))

cmf = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
cie_xyz_cmf = cmf[wavelengths].T.astype(np.float64)
illum_D65 = colour.SDS_ILLUMINANTS['D65']
d65_raw = illum_D65[wavelengths].astype(np.float64)
k_d65 = 100.0 / np.sum(d65_raw * cie_xyz_cmf[1])
d65_scaled = d65_raw * k_d65
xy_white_d65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

Z_train = np.load('results/global_cmf_reoptimization_20261028_205403/data/Z_train.npy')
H_train = np.maximum(b_sub[:, None] - A_sub @ Z_train.T, 1e-12)
pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
R_train = np.clip(pca_mean + Z_train @ pca_basis.T, 0.0, 1.0)
XYZ_train_ref = (R_train * d65_scaled) @ cie_xyz_cmf.T

bounds = [
    (0.2, 5.0), (420.0, 475.0), (15.0, 60.0),
    (0.2, 5.0), (515.0, 575.0), (20.0, 65.0),
    (0.2, 5.0), (590.0, 660.0), (20.0, 75.0),
    (0.0, 1.5), (420.0, 470.0), (10.0, 45.0)
]

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

def evaluate_ambiguity_fast_pre(M, H_pre):
    A_theta = M @ pca_basis
    U, S, Vt = np.linalg.svd(A_theta)
    if S[-1] < 1e-4:
        return {"spread_mean": 999.0, "spread_median": 999.0, "spread_p95": 999.0, "all_spreads": [999.0] * H_pre.shape[1]}
    N_theta = Vt[3:].T
    G = A_sub @ N_theta
    n_pts = H_pre.shape[1]
    spans = np.zeros((n_pts, 5))
    for j in range(5):
        G_j = G[:, j]
        p_mask = G_j > 1e-9
        t_p = np.min(H_pre[p_mask] / G_j[p_mask, None], axis=0)
        n_mask = G_j < -1e-9
        t_n = np.min(H_pre[n_mask] / (-G_j[n_mask, None]), axis=0)
        spans[:, j] = t_p + t_n
    spreads = np.sqrt(np.sum(spans ** 2, axis=1) / 12.0)
    return {
        "spread_mean": float(np.mean(spreads)),
        "spread_median": float(np.median(spreads)),
        "spread_p95": float(np.percentile(spreads, 95)),
        "all_spreads": spreads
    }

amb_base = evaluate_ambiguity_fast_pre(M_baseline, H_train)
old_spreads_train = np.array(amb_base["all_spreads"])

def evaluate_colorimetry_batch(M, R_spectra, XYZ_ref):
    C_cam = (R_spectra * d65_scaled) @ M.T
    T, _, _, _ = np.linalg.lstsq(C_cam, XYZ_ref, rcond=None)
    XYZ_est = np.maximum(C_cam @ T, 1e-5)
    XYZ_ref_c = np.maximum(XYZ_ref, 1e-5)
    Lab_ref = colour.XYZ_to_Lab(XYZ_ref_c, illuminant=xy_white_d65)
    Lab_est = colour.XYZ_to_Lab(XYZ_est, illuminant=xy_white_d65)
    de00 = colour.difference.delta_E_CIE2000(Lab_ref, Lab_est)
    de00 = np.nan_to_num(de00, nan=10.0, posinf=10.0, neginf=10.0)
    return {"mean_de00": float(np.mean(de00))}

def compute_smoothness_penalty(M):
    dl = 10.0
    c_smooth = 0.0
    for k in range(3):
        d2 = (M[k, 2:] - 2.0 * M[k, 1:-1] + M[k, :-2]) / (dl ** 2)
        c_smooth += float(np.sum(d2 ** 2) * dl)
    return c_smooth

def compute_conditioning(M):
    MMT = M @ M.T
    eigvals = np.linalg.eigvalsh(MMT)
    cond = float(np.sqrt(max(eigvals[-1] / max(eigvals[0], 1e-12), 1.0)))
    return cond

def joint_objective(params):
    M = params_to_matrix(params)
    amb = evaluate_ambiguity_fast_pre(M, H_train)
    if amb["spread_mean"] >= 900.0:
        return 1e6
    spreads = np.array(amb["all_spreads"])
    reg_penalty = float(np.mean(np.maximum(0.0, spreads - old_spreads_train)))
    J_amb = (0.45 * amb["spread_mean"] + 0.35 * amb["spread_median"] + 0.20 * amb["spread_p95"] + 0.15 * reg_penalty)
    col = evaluate_colorimetry_batch(M, R_train, XYZ_train_ref)
    pen_de = 0.02 * col["mean_de00"] + 5.0 * (max(col["mean_de00"] - 1.8, 0.0) ** 2)
    smoothness = compute_smoothness_penalty(M)
    cond = compute_conditioning(M)
    pen_cond = 0.01 * (max(cond - 12.0, 0.0) ** 2)
    pen_smooth = 5.0 * smoothness
    return float(J_amb + pen_smooth + pen_cond + pen_de)

def fit_matrix_to_params(M_target):
    b_peak = wavelengths[np.argmax(M_target[2])]
    g_peak = wavelengths[np.argmax(M_target[1])]
    r_peak = wavelengths[np.argmax(M_target[0])]
    p0 = [2.0, np.clip(b_peak, 430.0, 470.0), 25.0, 2.0, np.clip(g_peak, 520.0, 560.0), 35.0, 2.0, np.clip(r_peak, 600.0, 640.0), 40.0, 0.2, 445.0, 20.0]
    res = minimize(lambda p: np.sum((params_to_matrix(p) - M_target) ** 2), p0, bounds=bounds, method='L-BFGS-B')
    return res.x

exp_dir = r"c:\Users\saipu\OneDrive\Desktop\fiber polytopes-2\camera comparisions\reports\Experiment_4_Initialization_Independence_20261031_221323"
maxiter_cams = ["Point_Grey_Grasshopper", "Original_Baseline_Camera", "Random_Smooth_6_seed606", "Random_Smooth_7_seed707"]

print("Testing continuation of 4 MAXITER runs:")
for cam in maxiter_cams:
    m_path = os.path.join(exp_dir, "Final_Optimized_Curves", f"{cam}_optimized_sensitivity.npy")
    M_curr = np.load(m_path)
    p_curr = fit_matrix_to_params(M_curr)
    j_start = joint_objective(p_curr)
    
    t0 = time.time()
    res = minimize(joint_objective, p_curr, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-6})
    runtime = time.time() - t0
    
    j_end = res.fun
    delta_j = j_start - j_end
    rel_change = abs(delta_j) / max(abs(j_start), 1e-12)
    grad_norm = np.linalg.norm(res.jac) if hasattr(res, 'jac') and res.jac is not None else 0.0
    
    print(f"\nCamera: {cam}")
    print(f"  J_start = {j_start:.6f} -> J_end = {j_end:.6f}")
    print(f"  Absolute delta_J = {delta_j:.4e} (Rel change: {rel_change:.4%})")
    print(f"  Iterations: {res.nit}, Function Evals: {res.nfev}")
    print(f"  Termination: {res.message}")
    print(f"  Gradient norm: {grad_norm:.4e}")
    print(f"  Convergence status: {'CONVERGED (Stationary)' if rel_change < 1e-4 else 'IMPROVED'}")
