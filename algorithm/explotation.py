import itertools
from typing import List, Tuple
import numpy as np

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