import os
import numpy as np


class CameraModel:
    """
    Camera spectral sensitivity model M (C x W), mapping 31-dimensional spectral
    reflectance R (400-700 nm) to color observations c = M @ R.
    
    Supports:
    - Default calibrated camera (reconstructed from PCA projection artifacts)
    - Custom / candidate camera spectral sensitivity matrices M(theta) for camera comparison
    - Analytical channel models (e.g. Gaussian filters) for sensor design research
    """
    def __init__(self, M=None, wavelengths=None, name="Default Camera"):
        self.name = name
        if wavelengths is None:
            self.wavelengths = np.linspace(400, 700, 31)
        else:
            self.wavelengths = np.array(wavelengths, dtype=float)
            
        if M is not None:
            self.M = np.array(M, dtype=float)
            self._validate_dimensions()
        else:
            self.M = None

    def _validate_dimensions(self):
        if self.M.ndim != 2:
            raise ValueError(f"Measurement matrix M must be 2D, got shape {self.M.shape}")
        if self.M.shape[1] != len(self.wavelengths):
            raise ValueError(
                f"M has {self.M.shape[1]} wavelength bands, but expected {len(self.wavelengths)}"
            )

    @classmethod
    def from_artifacts(cls, artifacts_dir):
        """
        Loads or derives the default camera model from artifacts.
        Reconstructs M such that M @ B = A_camera and M @ mu = camera_offset.
        """
        m_path = os.path.join(artifacts_dir, "M_camera.npy")
        if os.path.exists(m_path):
            M = np.load(m_path)
        else:
            B = np.load(os.path.join(artifacts_dir, "pca_basis.npy"))
            mu = np.load(os.path.join(artifacts_dir, "pca_mean.npy"))
            A = np.load(os.path.join(artifacts_dir, "A_camera.npy"))
            offset = np.load(os.path.join(artifacts_dir, "camera_offset.npy"))
            X = np.hstack([B, mu[:, None]])
            Y = np.hstack([A, offset[:, None]])
            M_est, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
            M = M_est.T

        wl_path = os.path.join(artifacts_dir, "wavelengths.npy")
        wavelengths = np.load(wl_path) if os.path.exists(wl_path) else np.linspace(400, 700, 31)
        return cls(M=M, wavelengths=wavelengths, name="Default Calibrated Camera")

    @classmethod
    def create_synthetic_gaussian(cls, centers=(450.0, 540.0, 610.0), fwhm=50.0, wavelengths=None, name="Synthetic Gaussian Camera"):
        """
        Creates a candidate 3-channel camera with Gaussian spectral response curves.
        """
        if wavelengths is None:
            wavelengths = np.linspace(400, 700, 31)
        sigma = fwhm / 2.35482
        channels = []
        for mu_c in centers:
            resp = np.exp(-0.5 * ((wavelengths - mu_c) / sigma) ** 2)
            # Normalize channel peak to 1.0
            resp /= np.max(resp)
            channels.append(resp)
        M = np.array(channels)
        return cls(M=M, wavelengths=wavelengths, name=name)

    def project_pca(self, pca_basis, pca_mean):
        """
        Projects PCA model to measurement space:
        A = M @ B (C x K)
        b_offset = M @ mu (C,)
        """
        if self.M is None:
            raise ValueError("Measurement matrix M is not set.")
        A = self.M @ pca_basis
        b_offset = self.M @ pca_mean
        return A, b_offset

    def measure(self, spectra):
        """
        Measures spectral reflectance R to produce color observation:
        c = M @ R.
        Supports single spectrum (31,) or batch (N, 31).
        """
        if self.M is None:
            raise ValueError("Measurement matrix M is not set.")
        if spectra.ndim == 1:
            return self.M @ spectra
        else:
            return (self.M @ spectra.T).T

    @property
    def num_channels(self):
        return self.M.shape[0] if self.M is not None else 0
