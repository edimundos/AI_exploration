import numpy as np
from scipy.stats.qmc import Sobol, scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from config.config import (
    GLOBAL_EXPLORATION,
    LOCAL_EXPLORATION,
    SEED,
    TOP_K_REGIONS,
    LOCAL_WINDOW,
    TOTAL_EXPLORATION
)
from utils.utils import clip, query, query_int

def phase1_sobol(explorer, n: int = GLOBAL_EXPLORATION) -> None:
    # make some random points in the box using Sobol thing
    m = int(np.ceil(np.log2(n)))
    sobol = Sobol(d=2, scramble=True, seed=SEED)
    samples = sobol.random_base2(m)[:n]
    points = scale(samples, explorer.low, explorer.high)
    # try all the points, just loop and query
    for x, y in points:
        explorer.query(x, y)


def phase2_bayesian(explorer, n: int = LOCAL_EXPLORATION) -> None:
    # get only the points that are not nan, so we don't mess up
    X_arr = np.array(
        [(x, y) for (x, y), z in zip(explorer.X, explorer.y) if not np.isnan(z)]
    )
    y_arr = np.array([z for z in explorer.y if not np.isnan(z)])
    if len(y_arr) < 5:
        return  # not enough data, just skip

    # fit the GP model, it's like a smart guesser
    kernel = Matern(length_scale=20.0, nu=2.5)
    explorer.gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=SEED,
    ).fit(X_arr, y_arr)

    # pick the best K points, like top scores
    top_idx = np.argsort(y_arr)[-TOP_K_REGIONS:]
    centres = X_arr[top_idx]
    per_center = n // len(centres)
    rng = np.random.default_rng(SEED)

    # for each center, try to find better points around it
    for cx, cy in centres:
        for t in range(per_center):
            if t % 10 == 0:
                # sometimes just pick random nearby
                cand = rng.uniform(
                    low=[cx - LOCAL_WINDOW/2, cy - LOCAL_WINDOW/2],
                    high=[cx + LOCAL_WINDOW/2, cy + LOCAL_WINDOW/2],
                    size=(1,2),
                )[0]
            else:
                # most times, try 300 randoms and pick the best by EI
                cand = rng.uniform(
                    low=[cx - LOCAL_WINDOW/2, cy - LOCAL_WINDOW/2],
                    high=[cx + LOCAL_WINDOW/2, cy + LOCAL_WINDOW/2],
                    size=(300,2),
                )
                mu, sigma = explorer.gp.predict(cand, return_std=True)
                y_best = np.nanmax(y_arr)
                imp = mu - y_best
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
                cand = cand[np.argmax(ei)]
            # make sure it's inside the box
            x_next, y_next = clip(*cand, explorer.low, explorer.high)
            explorer.query(x_next, y_next)


def explore(explorer) -> None:
    # run both phases, first random then smart
    phase1_sobol(explorer)
    phase2_bayesian(explorer)
    # cut off extra points if too many, keep it under budget
    explorer.X = explorer.X[:TOTAL_EXPLORATION]
    explorer.y = explorer.y[:TOTAL_EXPLORATION]
