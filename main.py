import requests
import numpy as np
import matplotlib.pyplot as plt

# Base URL for exploration API
def query_api(x: float, y: float) -> float:
    """
    Query the exploration API for the value at coordinates (x, y).
    Parses the JSON response {"z": value} and returns the float.
    """
    url = f"http://157.180.73.240:8080/{x}/{y}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return float(data['z'])


def generate_grid_values(x_min: float, x_max: float,
                         y_min: float, y_max: float,
                         nx: int, ny: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Systematically sample the API on a grid.
    - x_min, x_max: range for x
    - y_min, y_max: range for y
    - nx, ny: number of points in x and y dimensions (nx*ny calls)

    Returns:
      X, Y: 2D meshgrid arrays of shape (ny, nx)
      Z: 2D array of sampled values from the API
    """
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    for i in range(ny):
        for j in range(nx):
            x_val = float(X[i, j])
            y_val = float(Y[i, j])
            try:
                Z[i, j] = query_api(x_val, y_val)
            except Exception as e:
                print(f"Error querying ({x_val}, {y_val}): {e}")
                Z[i, j] = np.nan
    return X, Y, Z


def plot_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Plot a heatmap of the sampled values.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(Z,
               origin='lower',
               extent=[X.min(), X.max(), Y.min(), Y.max()],
               aspect='auto'
               )
    plt.colorbar(label='API value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmap of API Responses')
    plt.show()


if __name__ == '__main__':
    # Systematic grid sampling with about 200 points
    nx, ny = 14, 14  # 14*14 = 196 calls
    X, Y, Z = generate_grid_values(-100, 100, -100, 100, nx, ny)
    plot_heatmap(X, Y, Z)
