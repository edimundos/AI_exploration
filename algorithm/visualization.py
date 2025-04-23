import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from config.config import RESULTS_DIR

def plot_exploration(explorer, filename=str(RESULTS_DIR / "exploration_results.png")) -> None:
    # get x, y, z from explorer, z is like the result
    xs, ys = zip(*explorer.X)
    zs = explorer.y

    # make a scatter plot, colors show z values
    fig, ax = plt.subplots(figsize=(10,9))
    sc = ax.scatter(xs, ys, c=zs, cmap="viridis", s=50, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="API Response (z)")

    # if enough points, draw a smooth background using interpolation
    if len(explorer.X) >= 50 and not all(np.isnan(zs)):
        xi = np.linspace(explorer.low, explorer.high, 100)
        yi = np.linspace(explorer.low, explorer.high, 100)
        zi = griddata(
            (np.array(xs), np.array(ys)),
            np.array(zs),
            (xi[None,:], yi[:,None]),
            method="cubic"
        )
        # this fills the plot with color blobs, looks cool
        ctf = ax.contourf(xi, yi, zi, levels=15, cmap="viridis", alpha=0.3)
        fig.colorbar(ctf, ax=ax, label="Interpolated")

    # add some labels and grid, save the pic
    ax.set_title("Exploration Results")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)


def plot_exploitation(explorer, filename=str(RESULTS_DIR / "exploitation_path.png")) -> None:
    # if there is no path, just quit
    if not explorer.exploitation_path:
        return

    # get x, y, z for the path
    xp, yp, zp = zip(*explorer.exploitation_path)
    fig, ax = plt.subplots(figsize=(9,9))

    # draw the path with circles and lines
    ax.plot(xp, yp, "o-", lw=2, ms=8, alpha=0.7)
    sc = ax.scatter(xp, yp, c=zp, cmap="plasma", s=160, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Value (z)")

    # label each point with its order and value, makes it easier to see
    for i,(x,y,z) in enumerate(explorer.exploitation_path):
        ax.text(x, y, f"{i+1}\n{z:.0f}", ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # add title, labels, grid, save it
    ax.set_title("10-Step Exploitation Path")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
