"""fine-tracing: turn crack masks into measured centrelines.

>>> from fine_tracing import trace
>>> result = trace(mask, image=photo, px_size=0.1)   # 0.1 mm per pixel
>>> for crack in result.cracks:
...     print(crack.id, crack.length, crack.width_stats()["max"])
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import TraceConfig  # noqa: E402
from .pipeline import trace  # noqa: E402
from .result import Crack, Path, TraceResult  # noqa: E402

__all__ = ["trace", "TraceConfig", "TraceResult", "Crack", "Path", "__version__"]

