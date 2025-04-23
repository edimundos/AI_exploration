
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats.qmc import Sobol, scale
from utils.utils import clip, call_api, query_int, save_csv

def phase1_sobol(n: int = GLOBAL_EXPLORATION):
    """Sobol space‑filling design in the box."""

    m = int(np.ceil(np.log2(n)))
    sobol = Sobol(d=2, scramble=True, seed=SEED)
    samples = sobol.random_base2(m)[:n]
    points = scale(samples, self.low, self.high)

    for x, y in points:
        query(float(x), float(y))

def phase2_bayesian(self, n: int = LOCAL_EXPLORATION):
    """Expected‑Improvement search in local windows around best points."""
    # Fit GP on phase‑1 data (skip NaNs)
    X_arr = np.array(
        [(x, y) for (x, y), z in zip(self.X, self.y) if not np.isnan(z)]
    )
    y_arr = np.array([z for z in self.y if not np.isnan(z)])
    if len(y_arr) < 5:  # defensive
        return

    kernel = Matern(length_scale=20.0, nu=2.5)
    self.gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=SEED,
    ).fit(X_arr, y_arr)

    # choose K peak centres
    top_idx = np.argsort(y_arr)[-TOP_K_REGIONS:]
    centres = X_arr[top_idx]

    samples_per_centre = n // len(centres)
    for centre in centres:
        cx, cy = centre
        for t in range(samples_per_centre):
            if t % 10 == 0:  # Thompson step
                cand = self.rng.uniform(
                    low=[cx - LOCAL_WINDOW / 2, cy - LOCAL_WINDOW / 2],
                    high=[cx + LOCAL_WINDOW / 2, cy + LOCAL_WINDOW / 2],
                    size=(1, 2),
                )
                x_next, y_next = self.clip(*cand[0])
            else:
                # EI maximisation via random sampling
                n_cand = 300
                cand = self.rng.uniform(
                    low=[cx - LOCAL_WINDOW / 2, cy - LOCAL_WINDOW / 2],
                    high=[cx + LOCAL_WINDOW / 2, cy + LOCAL_WINDOW / 2],
                    size=(n_cand, 2),
                )
                mu, sigmaboi = self.gp.predict(cand, return_std=True)
                y_best = np.nanmax(y_arr)
                with np.errstate(divide="ignore"):
                    imp = mu - y_best
                    Z = imp / sigmaboi
                    ei = imp * norm.cdf(Z) + sigmaboi * norm.pdf(Z)
                    ei[sigmaboi == 0.0] = 0.0
                x_next, y_next = cand[np.argmax(ei)]
                x_next, y_next = self.clip(float(x_next), float(y_next))
            self.query(x_next, y_next)

def explore():
    """Full 200‑step exploration procedure."""
    phase1_sobol()
    phase2_bayesian()

    # safety: keep only first TOTAL_EXPLORATION points
    self.X, self.y = self.X[:TOTAL_EXPLORATION], self.y[:TOTAL_EXPLORATION]
