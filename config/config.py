from typing import Tuple
import os

# Read in the env values and set default values if not provided
class Config:
    default_bounds: Tuple[int, int] = tuple(map(int, os.getenv("DEFAULT_BOUNDS", "-100,100").split(',')))
    global_exploration: int = int(os.getenv("GLOBAL_EXPLORATION", 50))
    local_exploration: int = int(os.getenv("LOCAL_EXPLORATION", 150))
    total_exploration: int = global_exploration + local_exploration
    top_k_regions: int = int(os.getenv("TOP_K_REGIONS", 4))
    local_window: float = float(os.getenv("LOCAL_WINDOW", 30.0))
    beam_width: int = int(os.getenv("BEAM_WIDTH", 7))
    path_steps: int = int(os.getenv("PATH_STEPS", 10))
    max_start_candidates: int = int(os.getenv("MAX_START_CANDIDATES", 30))
    seed: int = int(os.getenv("SEED", 42))
    base_url: str = os.getenv("BASE_URL", "http://157.180.73.240:8080")
