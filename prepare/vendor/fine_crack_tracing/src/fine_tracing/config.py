"""Settings for a tracing run.

Every setting has a default that works on masks from a few hundred to a few
thousand pixels wide. Lengths are in pixels unless the name says otherwise;
``px_size`` turns the reported lengths and widths into a physical unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Optional, Tuple

METHODS = ("auto", "skeleton", "corners")


@dataclass(frozen=True)
class TraceConfig:
    # ---- reading the mask -------------------------------------------------
    threshold: float = 0.5
    """Mask pixels at or above this fraction of full scale count as crack."""
    hysteresis: Optional[Tuple[float, float]] = None
    """(low, high) fractions for probability maps: weak pixels are kept only
    when they touch a strong one. Overrides ``threshold`` when set."""

    # ---- cleaning ---------------------------------------------------------
    min_area: int = 20
    """Mask components smaller than this many pixels are dropped as specks."""
    fill_holes: float = -1
    """Holes up to this many pixels are filled. -1 picks a size from the
    crack width; 0 turns hole filling off."""

    # ---- tracing ----------------------------------------------------------
    method: str = "auto"
    """``skeleton`` thins the mask to its centreline, ``corners`` places
    Shi-Tomasi corners, pulls them onto the centreline and links them with a
    minimum spanning tree, ``auto`` uses the skeleton and falls back to
    corners for any crack whose skeleton does not describe its mask."""
    prune: float = 2.5
    """Side branches shorter than this many local half-widths are cut."""
    prune_min: float = 3.0
    """...and side branches shorter than this many pixels always are."""
    bridge: bool = True
    """Join crack pieces that a segmentation break split apart."""
    bridge_max_gap: float = -1
    """Longest gap to join, px. -1 scales it with the crack: three widths of
    the two ends being joined, and never less than 30 px."""
    bridge_max_angle: float = 45.0
    """How far, in degrees, each end may turn to meet the other."""
    extend_ends: bool = True
    """Carry each free end of the skeleton out to the tip of the mask."""
    smoothing: float = 2.0
    """Gaussian smoothing along the path, px. 0 keeps the pixel staircase."""
    recentre: bool = True
    """Shift each point to the middle of the mask across the crack."""
    step: float = 1.0
    """Spacing of the output points, px."""
    min_length: float = 15.0
    """Cracks shorter than this, px, are dropped."""

    # ---- corner fallback --------------------------------------------------
    corner_quality: float = 0.02
    corner_min_distance: int = 5
    fallback_coverage: float = 0.8
    """In ``auto``, a skeleton that rebuilds less than this share of its mask
    hands the crack to the corner tracer."""

    # ---- units ------------------------------------------------------------
    px_size: Optional[float] = None
    """Size of one pixel in ``unit`` (for example 0.1 for 0.1 mm/px)."""
    unit: str = "mm"
    width_classes: Tuple[float, ...] = field(default_factory=tuple)
    """Width class edges in output units, e.g. (0.1, 0.3, 1.0): the report
    gives the crack length that falls in each class."""

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold is a fraction in (0, 1]")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.px_size is not None and self.px_size <= 0:
            raise ValueError("px_size must be positive")

    def with_(self, **changes: Any) -> "TraceConfig":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["width_classes"] = list(self.width_classes)
        if self.hysteresis is not None:
            d["hysteresis"] = list(self.hysteresis)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        d = dict(d)
        if d.get("hysteresis") is not None:
            d["hysteresis"] = tuple(d["hysteresis"])
        if "width_classes" in d:
            d["width_classes"] = tuple(d["width_classes"])
        return cls(**d)


__all__ = ["TraceConfig", "METHODS"]

