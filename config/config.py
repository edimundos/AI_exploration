import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file, if present
load_dotenv()

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_env_tuple_int(name: str, default: tuple[int, int]) -> tuple[int, int]:
    val = os.getenv(name)
    if val:
        parts = val.split(",")
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                pass
    return default

DEFAULT_BOUNDS: tuple[int, int] = _get_env_tuple_int("DEFAULT_BOUNDS", (-100, 100))
GLOBAL_EXPLORATION = _get_env_int("GLOBAL_EXPLORATION", 50)
LOCAL_EXPLORATION = _get_env_int("LOCAL_EXPLORATION", 150)
TOTAL_EXPLORATION = GLOBAL_EXPLORATION + LOCAL_EXPLORATION
TOP_K_REGIONS = _get_env_int("TOP_K_REGIONS", 4)
LOCAL_WINDOW = _get_env_float("LOCAL_WINDOW", 30.0)
BEAM_WIDTH = _get_env_int("BEAM_WIDTH", 7)
PATH_STEPS = _get_env_int("PATH_STEPS", 10)
MAX_START_CANDIDATES = _get_env_int("MAX_START_CANDIDATES", 30)
SEED = _get_env_int("SEED", 42)
BASE_URL = os.getenv("BASE_URL", "http://157.180.73.240:8080")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
