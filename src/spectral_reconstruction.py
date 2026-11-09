import os
import numpy as np


class SpectralReconstructor:
    """
    Reconstructs 31-dimensional spectral reflectance curves R from PCA latent coordinates z:
    
        R = mu + B @ z
        
    Supports single vectors (K,) and batched point clouds (N, K).
    """
    def __init__(self, artifacts_dir=None, pca_basis=None, pca_mean=None):
        if pca_basis is not None and pca_mean is not None:
            self.pca_basis = np.asarray(pca_basis, dtype=np.float64)
            self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
        elif artifacts_dir is not None:
            self.pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
            self.pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)
        else:
            raise ValueError("Must provide either artifacts_dir or (pca_basis, pca_mean)")

        self.num_bands = self.pca_basis.shape[0]
        self.dim = self.pca_basis.shape[1]

    def reconstruct(self, z):
        """
        Reconstructs the 31-dimensional spectra from PCA coordinates.
        z can be a single vector (K,) or an array of shape (N, K).
        Returns spectra of shape (31,) or (N, 31).
        """
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 1:
            return self.pca_mean + self.pca_basis @ z_arr
        else:
            return self.pca_mean + z_arr @ self.pca_basis.T
