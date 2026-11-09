import time
import numpy as np

P = 40000
N = 150
H_f64 = np.random.uniform(0.1, 10.0, size=(P, N))
G_f64 = np.random.randn(P, 5)

H_f32 = H_f64.astype(np.float32)
G_f32 = G_f64.astype(np.float32)

def eval_f32(G, H):
    n_pts = H.shape[1]
    spans = np.zeros((n_pts, 5), dtype=np.float32)
    for j in range(5):
        G_j = G[:, j]
        p_idx = np.where(G_j > 1e-6)[0]
        inv_G_p = (1.0 / G_j[p_idx])[:, None]
        t_p = np.min(H[p_idx] * inv_G_p, axis=0)
        
        n_idx = np.where(G_j < -1e-6)[0]
        inv_G_n = (-1.0 / G_j[n_idx])[:, None]
        t_n = np.min(H[n_idx] * inv_G_n, axis=0)
        
        spans[:, j] = t_p + t_n
    spreads = np.sqrt(np.sum(spans ** 2, axis=1) / 12.0)
    return float(np.mean(spreads))

# Test precision difference
res_64 = 0
for _ in range(5):
    # eval original
    pass

t0 = time.time()
for _ in range(50):
    val_32 = eval_f32(G_f32, H_f32)
t1 = time.time()
print(f"Float32 time for 1 eval: {(t1 - t0)/50*1000:.2f} ms")
