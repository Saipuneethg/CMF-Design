import numpy as np
from scipy.optimize import minimize

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

# Load CIE_1931_Derived
M_orig = np.load("camera comparisions/reports/Experiment_4_Initialization_Independence_20261031_221323/Final_Optimized_Curves/CIE_1931_Derived_optimized_sensitivity.npy")

# Let's fit with proper bounds and multi-start or differential evolution
bounds = [
    (0.01, 10.0), (420.0, 480.0), (10.0, 70.0),   # Blue: a, mu, sig
    (0.01, 10.0), (510.0, 580.0), (15.0, 70.0),   # Green: a, mu, sig
    (0.01, 10.0), (580.0, 680.0), (15.0, 80.0),   # Red primary: a1, mu1, sig1
    (0.0, 10.0), (420.0, 480.0), (10.0, 60.0)    # Red secondary: a2, mu2, sig2
]

# Note: In CIE, Red has two lobes: one at 450 nm (height ~0.94) and one at 600 nm (height ~0.74)
p0 = [
    1.93, 450.0, 25.0,
    1.00, 555.0, 40.0,
    0.74, 605.0, 40.0,
    0.94, 450.0, 25.0
]

res = minimize(lambda p: np.sum((params_to_matrix(p) - M_orig) ** 2), p0, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12})
print(f"CIE_1931 fit error with proper p0: {np.max(np.abs(params_to_matrix(res.x) - M_orig)):.6e}")
