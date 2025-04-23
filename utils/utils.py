import sys
import time
import json
import numpy as np
from typing import Sequence, Tuple
from pathlib import Path
import pandas as pd

def clip(x: float, y: float, low: float, high: float) -> tuple[float,float]:
    """Clip a coordinate pair to the inclusive square bounds."""
    return (float(np.clip(x, low, high)), float(np.clip(y, low, high)))

def _call_api(self, x: float, y: float) -> float:
    """Low-level HTTP GET wrapper with retry and error handling."""
    url = f"{self.base_url}/{x}/{y}"
    for attempt in (1,2):
        try:
            r = self.session.get(url, timeout=3)
            r.raise_for_status()
            return float(json.loads(r.text)["z"])
        except Exception as exc:
            if attempt == 2:
                print(f"API failure at ({x:.3f},{y:.3f}): {exc}", file=sys.stderr)
                return np.nan
            time.sleep(0.5)
    return np.nan

def query(self, x: float, y: float) -> float:
    x, y = clip(x, y, self.low, self.high)
    z = self._call_api(x, y)
    self.X.append((x,y))
    self.y.append(z)
    return z

def query_int(self, x: int, y: int) -> float:
    if (x,y) in self.value_cache:
        return self.value_cache[(x,y)]
    xf, yf = clip(float(x), float(y), self.low, self.high)
    z = self._call_api(xf, yf)
    self.value_cache[(x,y)] = z
    return z

def save_csv(path: Path, data: Sequence[Tuple]) -> None:
    pd.DataFrame(data).to_csv(path, index=False)
