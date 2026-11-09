import os
import numpy as np
from scipy.optimize import minimize, differential_evolution

wavelengths = np.linspace(400, 700, 31)
target_throughput = 10.672252177502214

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

    s_b = (s_b / np.sum(s_b)) * target_throughput
    s_g = (s_g / np.sum(s_g)) * target_throughput
    s_r = (s_r / np.sum(s_r)) * target_throughput
    return np.array([s_r, s_g, s_b], dtype=np.float64)

bounds = [
    (0.01, 10.0), (420.0, 480.0), (10.0, 70.0),   # Blue: a, mu, sig
    (0.01, 10.0), (510.0, 580.0), (15.0, 70.0),   # Green: a, mu, sig
    (0.01, 10.0), (580.0, 680.0), (15.0, 80.0),   # Red primary: a1, mu1, sig1
    (0.0, 10.0), (420.0, 480.0), (10.0, 60.0)    # Red secondary: a2, mu2, sig2
]

def robust_fit(M_target):
    # Determine blue, green, red peak
    b_curve = M_target[2]
    g_curve = M_target[1]
    r_curve = M_target[0]
    
    b_peak = wavelengths[np.argmax(b_curve)]
    g_peak = wavelengths[np.argmax(g_curve)]
    # Red primary peak (>550)
    red_high_mask = wavelengths >= 550
    wls_red_high = wavelengths[red_high_mask]
    r_peak_primary = wls_red_high[np.argmax(r_curve[red_high_mask])]
    
    # Red secondary peak (<500)
    red_low_mask = wavelengths <= 500
    wls_red_low = wavelengths[red_low_mask]
    r_peak_sec = wls_red_low[np.argmax(r_curve[red_low_mask])]
    
    a_b_init = np.max(b_curve)
    a_g_init = np.max(g_curve)
    a_r1_init = np.max(r_curve[red_high_mask])
    a_r2_init = np.max(r_curve[red_low_mask])
    
    candidates = [
        [a_b_init, b_peak, 25.0, a_g_init, g_peak, 35.0, a_r1_init, r_peak_primary, 40.0, a_r2_init, r_peak_sec, 20.0],
        [2.0, 450.0, 25.0, 2.0, 540.0, 35.0, 2.0, 620.0, 40.0, 0.2, 445.0, 20.0],
        [1.93, 450.0, 25.0, 1.00, 555.0, 40.0, 0.74, 605.0, 40.0, 0.94, 450.0, 25.0],
    ]
    
    best_res = None
    best_loss = 1e9
    for c in candidates:
        res = minimize(lambda p: np.sum((params_to_matrix(p) - M_target) ** 2),
                       c, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
            
    if best_loss > 1e-4:
        # Polish with differential evolution if needed
        de = differential_evolution(lambda p: np.sum((params_to_matrix(p) - M_target) ** 2),
                                    bounds=bounds, maxiter=50, seed=42)
        res = minimize(lambda p: np.sum((params_to_matrix(p) - M_target) ** 2),
                       de.x, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_loss:
            best_res = res
            
    return best_res.x, params_to_matrix(best_res.x)

curves_dir = "camera comparisions/reports/Experiment_4_Initialization_Independence_20261031_221323/Final_Optimized_Curves"
camera_names = [
    "Canon_1DMarkIII", "Nikon_D200", "Point_Grey_Grasshopper",
    "CIE_1931_Derived", "Original_Baseline_Camera",
    "Random_Smooth_1_seed101", "Random_Smooth_2_seed202",
    "Random_Smooth_3_seed303", "Random_Smooth_4_seed404",
    "Random_Smooth_5_seed505", "Random_Smooth_6_seed606",
    "Random_Smooth_7_seed707", "Random_Smooth_8_seed808"
]

all_params = {}
for name in camera_names:
    M_orig = np.load(os.path.join(curves_dir, f"{name}_optimized_sensitivity.npy"))
    p_fit, M_fit = robust_fit(M_orig)
    err = np.max(np.abs(M_fit - M_orig))
    all_params[name] = p_fit
    print(f"{name:30s} max fit err: {err:.4e}")
