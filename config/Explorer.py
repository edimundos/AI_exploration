
"""
API2DExplorer: orchestrates exploration, exploitation, visualization & I/O.
"""

from pathlib import Path
import numpy as np

from config.config import BASE_URL, RESULTS_DIR, SEED, DEFAULT_BOUNDS
from utils.utils import clip, _call_api, query, query_int, save_csv
from algorithm.exploration import explore
from algorithm.explotation import beam_search
from algorithm.visualization import plot_exploration, plot_exploitation

class API2DExplorer:
    def __init__(self):
        # set up some stuff like url, bounds, rng
        self.base_url = BASE_URL
        self.low, self.high = DEFAULT_BOUNDS
        self.rng = np.random.default_rng(SEED)

        # lists for points and values
        self.X: list[tuple[float,float]] = []
        self.y: list[float] = []
        self.value_cache: dict[tuple[int,int], float] = {}

        # gp is for model, path is for results
        self.gp = None
        self.exploitation_path: list[tuple[int,int,float]] = []

        # make a session for http requests
        self.session = __import__("requests").Session()

    # just linking these functions from utils
    clip = clip
    _call_api = _call_api
    query = query
    query_int = query_int

    def run(self) -> None:
        # make sure results dir exists, or make it
        RESULTS_DIR.mkdir(exist_ok=True)

        # runs explore, saves results, plots
        explore(self)
        save_csv(RESULTS_DIR / "explore_results.csv",
                 [(x,y,z) for (x,y),z in zip(self.X, self.y)])
        plot_exploration(self)

        # does beam search, saves, plots
        beam_search(self)
        save_csv(RESULTS_DIR / "exploit_results.csv", self.exploitation_path)
        plot_exploitation(self)

        # sum up rewards and print, kinda like final score
        total = np.nansum([z for _,_,z in self.exploitation_path])
        print(f"TOTAL 10-move reward: {total:.2f}")
