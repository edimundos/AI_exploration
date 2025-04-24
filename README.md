Made by Eduards Žeiris and Jēkabs Čudars

Results can be found in "Results" folder

## Purpose

This document explains the code structure and functionality of the **AI Exploration** project. The project sends queries to a two-dimensional API to perform two tasks:

1. **Explore** the entire square area from **–100 to 100** on both axes in exactly **200 calls**.
2. **Exploit** one small area in **10 integer moves**, each only **±1** step away from the last move.

The goal is to maximize the total value returned by the API.

---

## Quick Start

- remove .example from .env.example and set your needed parameters

```bash
pip install -r requirements.txt
python main.py
```

The script will:
* Write **explore_results.csv** with the 200 exploration points.
* Write **exploit_results.csv** with the 10 exploitation moves.
* Save two PNG images showing the exploration points and exploitation path.
* Print the sum of the 10 exploitation values.

---

## Project Structure

| File/Directory | Purpose |
|----------------|---------|
| **main.py** | Entry point for running the entire pipeline. |
| **config/config.py** | Loads environment variables and defines global constants. |
| **config/Explorer.py** | Implements a custom `API2DExplorer` class to manage exploration, exploitation, and I/O. |
| **utils/utils.py** | Provides helper functions like `clip`, `_call_api`, `query`, and `save_csv`. |
| **algorithm/exploration.py** | Implements the exploration phase using Sobol sampling and Bayesian optimization. |
| **algorithm/explotation.py** | Implements the exploitation phase using beam search to find the optimal path. |
| **algorithm/visualization.py** | Generates visualizations for exploration and exploitation results. |
| **results/explore_results.csv** | Contains the exploration points and their API responses. |
| **results/exploit_results.csv** | Contains the exploitation path and its API responses. |
| **.env** | Environment variables for runtime configuration (e.g., API URL, exploration parameters). |
| **.env.example** | Example `.env` file with default values for configuration. |
| **README.md** | This file, explaining the project structure and usage. |

---

## Required Packages

The project requires the following Python packages:

```
matplotlib
numpy
pandas
requests
scipy
scikit-learn
python-dotenv
```

Ensure you have Python 3.9+ installed.

---

## Algorithm Overview

### 1. Exploration Phase (200 API Calls)

| Step | Description | Purpose |
|------|-------------|---------|
| 1.1 | Generate 100 points using **Sobol sampling** to cover the square evenly. | Provides an initial overview of the area. |
| 1.2 | Store coordinates and values in arrays **X** and **y**. | Organizes data for further processing. |
| 1.3 | Fit a **Gaussian Process (GP)** model to the data. | Predicts values and uncertainties for unexplored points. |
| 1.4 | Select the top **K** points (e.g., 5) as centers for local refinement. | Focuses on promising regions. |
| 1.5 | Add 100 more points around the centers using **Expected Improvement (EI)** and random sampling. | Refines the search in high-value areas. |
| 1.6 | Update the GP model after each new point. | Keeps predictions accurate. |

### 2. Exploitation Phase (10 Integer Moves)

| Step | Description | Purpose |
|------|-------------|---------|
| 2.1 | Round exploration coordinates to integers and keep the best value for each cell. | Prepares data for integer-based pathfinding. |
| 2.2 | Identify starting points with at least one positive neighbor. | Ensures valid paths can be initiated. |
| 2.3 | Use **beam search** to find the optimal 10-step path. | Maximizes the total value of the path. |

### 3. Output and Visualization

* **explore_results.csv**: Contains `x, y, z` for all exploration points.
* **exploit_results.csv**: Contains `x, y, z` for the exploitation path.
* **exploration_results.png**: Scatter plot of exploration points with interpolated background.
* **exploitation_path.png**: Line plot of the 10-step path overlaid on the exploration background.

---

## Configuration

The project uses environment variables for configuration. These can be set in the `.env` file. Key variables include:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_BOUNDS` | Bounds for the exploration area. | `"-100,100"` |
| `GLOBAL_EXPLORATION` | Number of points for global exploration. | `50` |
| `LOCAL_EXPLORATION` | Number of points for local exploration. | `150` |
| `TOP_K_REGIONS` | Number of regions to refine in local exploration. | `4` |
| `LOCAL_WINDOW` | Size of the local search window. | `30.0` |
| `BEAM_WIDTH` | Beam width for exploitation search. | `7` |
| `PATH_STEPS` | Number of steps in the exploitation path. | `10` |
| `BASE_URL` | API base URL. | `"http://157.180.73.240:8080"` |

---

## Customization

You can modify the following parameters in the `.env` file or `config/config.py`:

* **Search Budget**: Adjust `GLOBAL_EXPLORATION` or `LOCAL_EXPLORATION`.
* **Local Window Size**: Change `LOCAL_WINDOW`.
* **Beam Width or Path Length**: Modify `BEAM_WIDTH` or `PATH_STEPS`.

