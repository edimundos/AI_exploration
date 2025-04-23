import json
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

def clip(self, x: float, y: float) -> Tuple[float, float]:
        """Clip a coordinate pair to the inclusive square bounds."""
        return float(np.clip(x, self.low, self.high)), float(
            np.clip(y, self.low, self.high)
        )

def _call_api(self, x: float, y: float) -> float:
    """Low‑level HTTP GET wrapper with retry and error handling."""
    url = f"{self.base_url}/{x}/{y}"
    for attempt in (1, 2):
        try:
            r = self.session.get(url, timeout=3)
            r.raise_for_status()
            return float(json.loads(r.text)["z"])
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"API failure at ({x:.3f},{y:.3f}): {exc}", file=sys.stderr)
                return np.nan
            time.sleep(0.5)
    return np.nan  # never reached

def query(self, x: float, y: float) -> float:
    """Query and record a (possibly new) floating coordinate."""
    x, y = self.clip(x, y)
    z = self._call_api(x, y)
    self.X.append((x, y))
    self.y.append(z)
    return z

def query_int(self, x: int, y: int) -> float:
    """Query an *integer* coordinate with caching."""
    if (x, y) in self.value_cache:
        return self.value_cache[(x, y)]

    x_f, y_f = self.clip(float(x), float(y))
    z = self._call_api(x_f, y_f)
    self.value_cache[(x, y)] = z
    return z

def save_csv(self, path: Path, data: Sequence[Tuple]) -> None:
    pd.DataFrame(data).to_csv(path, index=False)