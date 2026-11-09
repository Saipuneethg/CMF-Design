import os
import numpy as np
from scipy.optimize import minimize

wavelengths = np.linspace(400, 700, 31)
bounds = [
    (0.2, 5.0), (420.0, 475.0), (15.0, 60.0),   # Blue: a, mu, sig
    (0.2, 5.0), (515.0, 575.0), (20.0, 65.0),   # Green: a, mu, sig
    (0.2, 5.0), (590.0, 660.0), (20.0, 75.0),   # Red primary: a1, mu1, sig1
    (0.0, 1.5), (420.0, 470.0), (10.0, 45.0)    # Red secondary: a2, mu2, sig2
]

def params_to_matrix(params, target_throughput=10.672252177502214):
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

def fit_matrix_to_params(M_target):
    b_peak = wavelengths[np.argmax(M_target[2])]
    g_peak = wavelengths[np.argmax(M_target[1])]
    r_peak = wavelengths[np.argmax(M_target[0])]

    p0 = [
        2.0, np.clip(b_peak, 430.0, 470.0), 25.0,
        2.0, np.clip(g_peak, 520.0, 560.0), 35.0,
        2.0, np.clip(r_peak, 600.0, 640.0), 40.0,
        0.2, 445.0, 20.0
    ]

    def loss(p):
        M_fit = params_to_matrix(p)
        return np.sum((M_fit - M_target) ** 2)

    res = minimize(loss, p0, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-8})
    return res.x, res.fun

# Test fitting on Point_Grey_Grasshopper
exp_dir = r"c:\Users\saipu\OneDrive\Desktop\fiber polytopes-2\camera comparisions\reports\Experiment_4_Initialization_Independence_20261031_221323"
m_path = os.path.join(exp_dir, "Final_Optimized_Curves", "Point_Grey_Grasshopper_optimized_sensitivity.npy")
M_pg = np.load(m_path)
p_fit, err = fit_matrix_to_params(M_pg)
print(f"Point Grey recovery squared error: {err:.4e}")
print("Recovered parameters:", p_fit)
