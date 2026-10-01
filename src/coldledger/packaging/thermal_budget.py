from dataclasses import dataclass
from datetime import datetime
from ..core.timeseries import Sample as Point, band_severity, integrate


@dataclass(frozen=True)
class Sample:
    at: datetime
    celsius: float


def exposure(samples, low, high, *, max_gap_seconds=None):
    """Time-integrated thermal exposure in severity-hours.

    Uses the shared trapezoidal integral (coldledger.core.timeseries) so
    budgets, sampling trajectories and deviation cases agree on the same
    series. Gaps wider than max_gap_seconds are rejected as sparse data
    rather than interpolated over.
    """
    points = [Point(s.at, s.celsius) for s in samples]
    return integrate(points, band_severity(low, high), max_gap_seconds=max_gap_seconds) / 3600.0
