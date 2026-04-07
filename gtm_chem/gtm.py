"""
Generative Topographic Mapping (GTM) — core implementation.

Reference:
    Bishop, C.M. & Svensén, M. (1998). GTM: The Generative Topographic Mapping.
    Neural Computation, 10(1), 215–234.

Key invariance property:
    Once the model is trained (W, β fixed), projecting NEW data onto the map
    is a pure E-step computation. The latent grid positions are NEVER updated,
    guaranteeing that the original chemical space topology is preserved when
    adding new molecules.
"""

from __future__ import annotations

import numpy as np
import pickle
import warnings
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


class GTM:
    """
    Generative Topographic Mapping.

    The model is a constrained mixture of Gaussians in data space:

        p(t | W, β) = (1/K) Σ_k  𝒩(t | y(u_k, W), β⁻¹ I)

    where y(u_k, W) = W · Φ(u_k)  is a smooth manifold parameterized by
    an (M+1 × D) weight matrix W and an RBF design matrix Φ.

    Training uses EM:
        E-step  → posterior responsibilities  R[n,k] = p(u_k | t_n, W, β)
        M-step  → update W (regularized least squares) and β (precision)

    Parameters
    ----------
    grid_size : int
        Number of nodes along each axis of the 2-D latent grid.
        Total latent nodes K = grid_size².  Recommended: 16–30.
    rbf_size : int
        Number of RBF centres along each axis.
        Total basis functions M = rbf_size² (+1 bias).  Recommended: 4–8.
    rbf_width : float or None
        σ of the Gaussian RBFs. None → auto (RBF spacing × scale_factor).
    rbf_width_scale : float
        Multiplier on the auto-computed σ.  >1 → smoother map.
    regularization : float
        λ — L2 weight penalty in the M-step linear solve. Prevents overfitting.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on log-likelihood change.
    verbose : bool
        Print progress during training.
    """

    def __init__(
        self,
        grid_size: int = 20,
        rbf_size: int = 5,
        rbf_width: float | None = None,
        rbf_width_scale: float = 1.0,
        regularization: float = 0.1,
        n_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        self.grid_size = grid_size
        self.rbf_size = rbf_size
        self.rbf_width = rbf_width
        self.rbf_width_scale = rbf_width_scale
        self.regularization = regularization
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hyperparameters(self):
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2.")
        if self.rbf_size < 1:
            raise ValueError("rbf_size must be >= 1.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1.")
        if self.tol <= 0:
            raise ValueError("tol must be > 0.")
        if self.regularization < 0:
            raise ValueError("regularization must be >= 0.")
        if self.rbf_width is not None and self.rbf_width <= 0:
            raise ValueError("rbf_width must be > 0 when provided.")
        if self.rbf_width is None and self.rbf_width_scale <= 0:
            raise ValueError("rbf_width_scale must be > 0.")

    def _check_input_matrix(self, X: np.ndarray, *, fitted: bool):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape={X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        if fitted and X.shape[1] != self.W_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects {self.W_.shape[1]}."
            )
        return X

    def _make_2d_grid(self, n: int) -> np.ndarray:
        """Regular grid of n² points in [–1, 1]²."""
        x = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _rbf_design_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Compute design matrix Φ for arbitrary latent points.
        Shape: (len(points), M + 1)  — M RBFs + 1 bias column.
        """
        sq_dists = cdist(points, self.rbf_centres_, "sqeuclidean")
        Phi = np.exp(-sq_dists / (2.0 * self.sigma_**2))
        return np.hstack([Phi, np.ones((len(points), 1))])

    def _responsibilities(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        E-step: compute posterior responsibilities and per-sample log-likelihoods.

        Returns
        -------
        R   : (N, K) responsibility matrix
        llk : (N,)  per-sample log p(t_n | W, β)
        """
        D = X.shape[1]
        Y = self.Phi_ @ self.W_          # (K, D)  manifold in data space
        sq = cdist(X, Y, "sqeuclidean")  # (N, K)

        # log p(t_n | u_k) = (D/2) log(β/2π) − (β/2) ‖t_n − y_k‖²
        log_p = 0.5 * D * np.log(self.beta_ / (2.0 * np.pi)) - 0.5 * self.beta_ * sq

        # log p(t_n | W, β) = log(1/K) + logsumexp_k log_p[n,k]
        log_mix = logsumexp(log_p, axis=1)          # (N,)
        llk = log_mix - np.log(self.K_)

        R = np.exp(log_p - log_mix[:, None])        # (N, K)
        return R, llk

    def _init_weights_pca(self, X: np.ndarray):
        """
        Initialise W by aligning the latent grid with the first two PCs of X.
        This drastically reduces the number of EM iterations needed.
        """
        n_components = min(2, X.shape[0], X.shape[1])
        mean = X.mean(axis=0)
        Xc = X - mean

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(X)
            components = pca.components_
            scale = np.sqrt(pca.explained_variance_)
        except Exception:
            # Fallback when sklearn is unavailable/incompatible:
            # principal axes from SVD on centred data.
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            components = Vt[:n_components]
            denom = max(X.shape[0] - 1, 1)
            scale = S[:n_components] / np.sqrt(denom)

        # Target positions: latent grid projected onto PC space
        latent = self.latent_grid_[:, :n_components]
        Y_init = latent @ np.diag(scale) @ components
        Y_init += mean[None, :]  # (K, D)

        # Solve  Φ W ≈ Y_init  in least-squares sense
        self.W_ = np.linalg.lstsq(self.Phi_, Y_init, rcond=None)[0]  # (M+1, D)

        # Initialise β from reconstruction error
        resid = Y_init - self.Phi_ @ self.W_
        var = np.mean(resid**2)
        self.beta_ = float(X.shape[1] / max(var, 1e-8))
        self.beta_ = min(self.beta_, 1e4)   # cap to avoid numerical issues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GTM":
        """
        Train GTM on data matrix X.

        Parameters
        ----------
        X : (N, D) float array — typically PCA-reduced or raw fingerprints.

        Returns
        -------
        self
        """
        self._validate_hyperparameters()
        X = self._check_input_matrix(X, fitted=False)
        N, D = X.shape
        if N < self.grid_size**2:
            warnings.warn(
                f"Dataset size ({N}) < number of latent nodes ({self.grid_size**2}). "
                "Consider reducing grid_size.",
                UserWarning,
                stacklevel=2,
            )

        # --- Build latent structures ---
        self.latent_grid_ = self._make_2d_grid(self.grid_size)  # (K, 2)
        self.K_ = len(self.latent_grid_)

        self.rbf_centres_ = self._make_2d_grid(self.rbf_size)   # (M, 2)
        self.M_ = len(self.rbf_centres_)

        if self.rbf_width is None:
            step = 2.0 / max(self.rbf_size - 1, 1)
            self.sigma_ = step * self.rbf_width_scale
        else:
            self.sigma_ = self.rbf_width

        # Φ on the fixed latent grid — shape (K, M+1)
        self.Phi_ = self._rbf_design_matrix(self.latent_grid_)

        # Regularisation matrix A — penalise weights, spare the bias
        self.A_ = np.eye(self.M_ + 1)
        self.A_[-1, -1] = 0.0

        # --- Initialise ---
        self._init_weights_pca(X)

        history = []
        llk_prev = -np.inf

        if self.verbose:
            print(f"Training GTM: N={N}, D={D}, K={self.K_}, M={self.M_+1}")
            print(f"  β_init={self.beta_:.4f}, σ_rbf={self.sigma_:.4f}")
            print("-" * 55)

        for it in range(self.n_iter):
            # ---- E-step ----
            R, llk_vec = self._responsibilities(X)          # (N, K), (N,)
            llk = float(np.mean(llk_vec))

            history.append(llk)
            if self.verbose and (it % 20 == 0 or it == self.n_iter - 1):
                print(f"  iter {it+1:4d}  mean-log-lik = {llk:.6f}")

            if abs(llk - llk_prev) < self.tol and it > 0:
                if self.verbose:
                    print(f"  Converged at iter {it+1}  (Δllk = {abs(llk - llk_prev):.2e})")
                break
            llk_prev = llk

            # ---- M-step ----
            # G = diag(Σ_n R[n,k])
            G = R.sum(axis=0)          # (K,)

            # (Φᵀ G Φ + λ/β A) W = Φᵀ Rᵀ X
            lhs = self.Phi_.T @ (self.Phi_ * G[:, None]) + \
                  (self.regularization / self.beta_) * self.A_
            rhs = self.Phi_.T @ (R.T @ X)              # (M+1, D)

            self.W_ = np.linalg.solve(lhs + 1e-10 * np.eye(self.M_ + 1), rhs)

            # Update β
            Y = self.Phi_ @ self.W_
            sq = cdist(X, Y, "sqeuclidean")            # (N, K)
            self.beta_ = float(N * D / np.sum(R * sq).clip(1e-12))
            self.beta_ = min(self.beta_, 1e6)

        self.log_likelihood_history_ = np.array(history)
        self.is_fitted = True
        if self.verbose:
            print(f"\n  Final β = {self.beta_:.4f}")
        return self

    def transform(
        self, X: np.ndarray, projection: str = "mean"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project data onto the 2-D latent space.

        This is a pure E-step — W and β are NEVER modified, preserving
        the topology of the original training space.

        Parameters
        ----------
        X          : (N, D) data to project.
        projection : 'mean'  → Σ_k R[n,k] u_k  (smooth, recommended)
                     'mode'  → argmax_k R[n,k]   (snaps to grid nodes)

        Returns
        -------
        coords : (N, 2) latent coordinates.
        R      : (N, K) responsibilities (useful for uncertainty).
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        X = self._check_input_matrix(X, fitted=True)
        R, _ = self._responsibilities(X)

        if projection == "mean":
            coords = R @ self.latent_grid_
        elif projection == "mode":
            coords = self.latent_grid_[np.argmax(R, axis=1)]
        else:
            raise ValueError(f"Unknown projection '{projection}'. Use 'mean' or 'mode'.")

        return coords, R

    def fit_transform(
        self, X: np.ndarray, projection: str = "mean"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and immediately transform the training data."""
        return self.fit(X).transform(X, projection)

    def landscape(self, X: np.ndarray) -> np.ndarray:
        """
        Data density landscape on the latent grid.

        Returns mean log-likelihood per latent node, shaped (grid_size, grid_size).
        High values = dense / high-activity regions.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        X = self._check_input_matrix(X, fitted=True)
        D = X.shape[1]
        Y = self.Phi_ @ self.W_
        sq = cdist(X, Y, "sqeuclidean")     # (N, K)
        log_p = 0.5 * D * np.log(self.beta_ / (2 * np.pi)) - 0.5 * self.beta_ * sq
        # Mean log-likelihood per node: logsumexp over molecules, minus log N
        node_llk = logsumexp(log_p, axis=0) - np.log(len(X))   # (K,)
        return node_llk.reshape(self.grid_size, self.grid_size)

    def node_activity(self, X: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Property/activity landscape on the latent grid.

        Weights each molecule's property by its responsibility,
        producing a smooth interpolated map.

        Parameters
        ----------
        X      : (N, D) molecules
        values : (N,)  any property (pIC50, MW, etc.)

        Returns
        -------
        (grid_size, grid_size) array of responsibility-weighted mean values.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        X = self._check_input_matrix(X, fitted=True)
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(X):
            raise ValueError(
                f"values length ({len(values)}) must match number of rows in X ({len(X)})."
            )

        R, _ = self._responsibilities(X)           # (N, K)
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.full((self.grid_size, self.grid_size), np.nan, dtype=float)

        Rv = R[valid]
        vv = values[valid]
        w_sum = Rv.sum(axis=0).clip(1e-12)         # (K,)
        weighted = (Rv * vv[:, None]).sum(axis=0) / w_sum
        return weighted.reshape(self.grid_size, self.grid_size)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Per-molecule projection uncertainty as entropy of the responsibility distribution.
        High entropy → molecule projects diffusely; low entropy → well-localised.

        Returns (N,) entropy values in nats.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        X = self._check_input_matrix(X, fitted=True)
        R, _ = self._responsibilities(X)
        R_safe = np.clip(R, 1e-30, None)
        entropy = -np.sum(R_safe * np.log(R_safe), axis=1)
        return entropy

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Pickle the fitted model."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ GTM model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "GTM":
        """Load a previously saved GTM model with NumPy compatibility."""
        
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a GTM instance.")
        print(f"✓ GTM model loaded ← {path}")
        return model
