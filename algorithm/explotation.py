import itertools
import numpy as np

from config.config import BEAM_WIDTH, PATH_STEPS, MAX_START_CANDIDATES
from utils.utils import query_int

def neighbours(x: int, y: int) -> list[tuple[int,int]]:
    # get all 8 cells around (x, y), not including itself
    return [
        (x+dx, y+dy)
        for dx, dy in itertools.product((-1,0,1), repeat=2)
        if not (dx == dy == 0)  # skip the center cell
    ]

def find_start_points(explorer) -> list[tuple[int,int]]:
    # collect points and keep the best z value for each int cell
    int_points: dict[tuple[int,int], float] = {}
    for (xf,yf), z in zip(explorer.X, explorer.y):
        xi, yi = int(round(xf)), int(round(yf))
        # only keep if it's better than before
        if ((xi,yi) not in int_points) or (z > int_points[(xi,yi)]):
            int_points[(xi,yi)] = z

    candidates = []
    for (xi, yi), z in int_points.items():
        if np.isnan(z): continue  # skip if z is not a number
        neigh = neighbours(xi, yi)
        # check if any neighbour is good (v > 0)
        vals = [query_int(explorer, nx, ny) for nx, ny in neigh]
        if any(v > 0 for v in vals if not np.isnan(v)):
            candidates.append((xi, yi, z))

    # sort by z value, biggest first
    candidates.sort(key=lambda t: t[2], reverse=True)
    # only keep top ones
    return [(x,y) for x,y,_ in candidates[:MAX_START_CANDIDATES]]


def beam_search(explorer) -> list[tuple[int,int,float]]:
    # find good starting points
    starts = find_start_points(explorer)
    best_value = -np.inf
    best_path: list[tuple[int,int]] = []

    for sx, sy in starts:
        start_val = query_int(explorer, sx, sy)
        beam = [(start_val, [(sx,sy)])]
        # try to build a path step by step
        for _ in range(1, PATH_STEPS):
            new_beam = []
            for value, path in beam:
                x,y = path[-1]
                # look at all neighbours
                for nx, ny in neighbours(x,y):
                    if not (explorer.low <= nx <= explorer.high): continue  # skip if out of bounds
                    if (nx,ny) in path: continue  # don't go back to same cell
                    nv = value + query_int(explorer, nx, ny)
                    new_beam.append((nv, path+[(nx,ny)]))
            if not new_beam: break  # stop if no more moves
            # keep only best few paths
            new_beam.sort(key=lambda t: t[0], reverse=True)
            beam = new_beam[:BEAM_WIDTH]

        # update best path if found something better
        if beam and beam[0][0] > best_value:
            best_value, best_path = beam[0]

    # make final path with values
    final = [(x,y, query_int(explorer, x,y)) for x,y in best_path]
    explorer.exploitation_path = final
    return final

