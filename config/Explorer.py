from typing import Dict, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
import requests
import numpy as np
import random

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