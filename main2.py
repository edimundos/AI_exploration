#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration–Exploitation engine for the 2‑D API challenge.
Authors : <YOUR NAMES HERE>
Course   : Advanced AI 2025 – Final Project
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import griddata
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats.qmc import Sobol, scale


import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------#
#                           CONFIGURATION SECTION                            #
# ---------------------------------------------------------------------------#

DEFAULT_BOUNDS: Tuple[int, int] = (-100, 100)
GLOBAL_EXPLORATION = 50          # phase‑1 samples
LOCAL_EXPLORATION = 150           # phase‑2 samples
TOTAL_EXPLORATION = GLOBAL_EXPLORATION + LOCAL_EXPLORATION
TOP_K_REGIONS = 4                 # refine this many peaks
LOCAL_WINDOW = 30.0               # square window side length in local phase
BEAM_WIDTH = 7
PATH_STEPS = 10
MAX_START_CANDIDATES = 30
SEED = 42                         # reproducibility
BASE_URL = "http://157.180.73.240:8080"

# ---------------------------------------------------------------------------#
#                               CORE CLASS                                   #
# ---------------------------------------------------------------------------#


class API2DExplorer:
    """Explore and exploit a bounded 2‑D real domain under query budget."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        bounds: Tuple[int, int] = DEFAULT_BOUNDS,
        seed: int | None = None,
    ):
        self.base_url = base_url
        self.bounds = bounds
        self.low, self.high = bounds
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.X: List[Tuple[float, float]] = []
        self.y: List[float] = []
        self.value_cache: Dict[Tuple[int, int], float] = {}

        self.gp: GaussianProcessRegressor | None = None
        self.exploitation_path: List[Tuple[int, int, float]] = []

        # http session for connection reuse
        self.session = requests.Session()

    # -------------------------  UTILITY HELPERS  --------------------------- #

    def clip(self, x: float, y: float) -> Tuple[float, float]:
        """Clip a coordinate pair to the inclusive square bounds."""
        return float(np.clip(x, self.low, self.high)), float(
            np.clip(y, self.low, self.high)
        )

    def _call_api(self, x: float, y: float) -> float:
        """Low‑level HTTP GET wrapper with retry and error handling."""
        url = f"{self.base_url}/{x}/{y}"
        for attempt in (1, 2):
            try:
                r = self.session.get(url, timeout=3)
                r.raise_for_status()
                return float(json.loads(r.text)["z"])
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"API failure at ({x:.3f},{y:.3f}): {exc}", file=sys.stderr)
                    return np.nan
                time.sleep(0.5)
        return np.nan  # never reached

    def query(self, x: float, y: float) -> float:
        """Query and record a (possibly new) floating coordinate."""
        x, y = self.clip(x, y)
        z = self._call_api(x, y)
        self.X.append((x, y))
        self.y.append(z)
        return z

    def query_int(self, x: int, y: int) -> float:
        """Query an *integer* coordinate with caching."""
        if (x, y) in self.value_cache:
            return self.value_cache[(x, y)]

        x_f, y_f = self.clip(float(x), float(y))
        z = self._call_api(x_f, y_f)
        self.value_cache[(x, y)] = z
        return z

    # -------------------------  EXPLORATION  ------------------------------- #

    def phase1_sobol(self, n: int = GLOBAL_EXPLORATION) -> None:
        """Sobol space‑filling design in the box."""

        m = int(np.ceil(np.log2(n)))
        sobol = Sobol(d=2, scramble=True, seed=SEED)
        samples = sobol.random_base2(m)[:n]
        points = scale(samples, self.low, self.high)

        for x, y in points:
            self.query(float(x), float(y))

    def phase2_bayesian(self, n: int = LOCAL_EXPLORATION) -> None:
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

    def explore(self) -> None:
        """Full 200‑step exploration procedure."""
        self.phase1_sobol()
        self.phase2_bayesian()

        # safety: keep only first TOTAL_EXPLORATION points
        self.X, self.y = self.X[:TOTAL_EXPLORATION], self.y[:TOTAL_EXPLORATION]

    # -------------------------  EXPLOITATION  ------------------------------ #

    @staticmethod
    def neighbours(x: int, y: int) -> List[Tuple[int, int]]:
        """8‑connected neighbours of an integer cell."""
        return [
            (x + dx, y + dy)
            for dx, dy in itertools.product((-1, 0, 1), repeat=2)
            if not (dx == dy == 0)
        ]

    def find_start_points(self) -> List[Tuple[int, int]]:
        """Return good integer start points sorted by cache value."""
        # convert exploration set to integer grid points
        int_points = {}
        for (xf, yf), z in zip(self.X, self.y):
            xi, yi = int(round(xf)), int(round(yf))
            if (xi, yi) not in int_points or z > int_points[(xi, yi)]:
                int_points[(xi, yi)] = z

        # evaluate neighbourhood positivity
        candidates = []
        for (xi, yi), z in int_points.items():
            if np.isnan(z):
                continue
            neigh_vals = [
                self.query_int(nx, ny) for nx, ny in self.neighbours(xi, yi)
            ]
            if any(val > 0 for val in neigh_vals if not np.isnan(val)):
                candidates.append((xi, yi, z))
        candidates.sort(key=lambda t: t[2], reverse=True)
        return [(x, y) for x, y, _ in candidates[:MAX_START_CANDIDATES]]

    def beam_search(self) -> List[Tuple[int, int, float]]:
        """Beam search of width BEAM_WIDTH for a 10‑step high‑sum path."""
        starts = self.find_start_points()
        best_value = -np.inf
        best_path: List[Tuple[int, int]] = []

        for sx, sy in starts:
            start_val = self.query_int(sx, sy)
            beam: List[Tuple[float, List[Tuple[int, int]]]] = [
                (start_val, [(sx, sy)])
            ]

            for _step in range(1, PATH_STEPS):
                new_beam = []
                for value, path in beam:
                    x, y = path[-1]
                    for nx, ny in self.neighbours(x, y):
                        if not (self.low <= nx <= self.high and self.low <= ny <= self.high):
                            continue
                        if (nx, ny) in path:
                            continue
                        new_val = value + self.query_int(nx, ny)
                        new_beam.append((new_val, path + [(nx, ny)]))
                if not new_beam:  # dead end
                    break
                new_beam.sort(key=lambda t: t[0], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            if beam and beam[0][0] > best_value:
                best_value, best_path = beam[0]

        # materialise z values
        final_path = [(x, y, self.query_int(x, y)) for x, y in best_path]
        self.exploitation_path = final_path
        return final_path

    # -------------------------  VISUALISATION  ----------------------------- #

    def plot_exploration(self, filename: str = "results/exploration_results.png") -> None:
        """Save a scatter+contour figure of exploration data."""
        xs, ys = zip(*self.X)
        zs = self.y
        fig, ax = plt.subplots(figsize=(10, 9))
        sc = ax.scatter(xs, ys, c=zs, cmap="viridis", s=50, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="API Response (z)")

        if len(self.X) >= 50 and not all(np.isnan(zs)):
            xi = np.linspace(self.low, self.high, 100)
            yi = np.linspace(self.low, self.high, 100)
            zi = griddata(
                (np.array(xs), np.array(ys)),
                np.array(zs),
                (xi[None, :], yi[:, None]),
                method="cubic",
            )
            ctf = ax.contourf(
                xi, yi, zi, levels=15, cmap="viridis", alpha=0.3
            )
            fig.colorbar(ctf, ax=ax, label="Interpolated")

        ax.set_title("Exploration Results")
        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)

    def plot_exploitation(self, filename: str = "results/exploitation_path.png") -> None:
        """Save a path plot of the exploitation moves."""
        if not self.exploitation_path:
            return
        xp, yp, zp = zip(*self.exploitation_path)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(xp, yp, "o-", lw=2, ms=8, alpha=0.7)
        sc = ax.scatter(xp, yp, c=zp, cmap="plasma", s=160, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Value (z)")
        for i, (x, y, z) in enumerate(self.exploitation_path):
            ax.text(
                x,
                y,
                f"{i+1}\n{z:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )
        ax.set_title("10‑Step Exploitation Path")
        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)

    # ------------------------------  I/O  ---------------------------------- #

    def save_csv(self, path: Path, data: Sequence[Tuple]) -> None:
        pd.DataFrame(data).to_csv(path, index=False)

    # ----------------------------  PIPELINE  -------------------------------- #

    def run(self) -> None:
        """Execute exploration → exploitation → plots."""
        self.explore()
        self.save_csv(Path("results/explore_results.csv"), [(x, y, z) for (x, y), z in zip(self.X, self.y)])

        self.plot_exploration()

        self.beam_search()
        self.save_csv(Path("results/exploit_results.csv"), self.exploitation_path)
        self.plot_exploitation()

        total = np.nansum([z for _, _, z in self.exploitation_path])
        print(f"TOTAL 10‑move reward: {total:.2f}")


# ---------------------------------------------------------------------------#
#                               ENTRY POINT                                  #
# ---------------------------------------------------------------------------#

def main() -> None:
    explorer = API2DExplorer(seed=SEED)
    explorer.run()


if __name__ == "__main__":
    main()
