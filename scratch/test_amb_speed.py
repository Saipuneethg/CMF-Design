import time
import numpy as np

# Simulate G and H
P = 40000
N = 150  # train set size (used in Experiment 4 for optimization and valley interpolation!)
H = np.random.uniform(0.1, 10.0, size=(P, N))
G = np.random.randn(P, 5)

def eval_original(G, H):
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
    return np.mean(spreads)

# Optimized version:
# Notice that H / G_j[:, None] = H * (1.0 / G_j)[:, None]
# and dividing P rows can be done in-place or without masking large arrays if we use masked min or positive values
def eval_fast(G, H):
    n_pts = H.shape[1]
    spans = np.zeros((n_pts, 5))
    for j in range(5):
        G_j = G[:, j]
        p_idx = np.where(G_j > 1e-9)[0]
        inv_G_p = (1.0 / G_j[p_idx])[:, None]
        t_p = np.min(H[p_idx] * inv_G_p, axis=0)
        
        n_idx = np.where(G_j < -1e-9)[0]
        inv_G_n = (-1.0 / G_j[n_idx])[:, None]
        t_n = np.min(H[n_idx] * inv_G_n, axis=0)
        
        spans[:, j] = t_p + t_n
    spreads = np.sqrt(np.sum(spans ** 2, axis=1) / 12.0)
    return np.mean(spreads)

t0 = time.time()
for _ in range(50):
    eval_original(G, H)
t1 = time.time()
print(f"Original for N=150: {(t1 - t0)/50*1000:.2f} ms")

t0 = time.time()
for _ in range(50):
    eval_fast(G, H)
t1 = time.time()
print(f"Fast for N=150: {(t1 - t0)/50*1000:.2f} ms")
