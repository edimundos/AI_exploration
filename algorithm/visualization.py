import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

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