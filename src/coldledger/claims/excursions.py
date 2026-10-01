"""Excursion detection on temperature sampling trajectories.

Detected excursions carry the same time-integrated severity used by the
packaging exposure budget (:mod:`coldledger.packaging.thermal_budget`), so a
deviation case and its budget consumption are always scored with one
integral. Sparse gaps are rejected and duplicate/replayed samples are
reconciled by the shared core.
"""
from dataclasses import dataclass
from datetime import datetime

from ..core import thermal


@dataclass(frozen=True)
class Sample:
    at: datetime
    value: float


@dataclass(frozen=True)
class Excursion:
    start: datetime
    end: datetime
    peak: float
    severity_integral: float = 0.0


def detect(samples, low, high, margin, min_seconds, max_gap_seconds=None):
    """Detect out-of-band excursions on a (possibly multi-segment) trajectory.

    Cross-segment replay and repeated samples are merged by the shared
    normalizer: identical samples at one instant are counted once, conflicting
    ones raise ``ValueError``. Each emitted excursion is qualified by duration
    and carries its shared severity*hours integral.
    """
    rows = thermal.normalize_rows(samples, max_gap_seconds)
    out = []
    start = None
    peak = None
    acc = thermal.SeverityAccumulator(low, high, max_gap_seconds)

    for t, value in rows:
        outside = value < low or value > high
        if start is None and outside:
            start = t
            peak = value
            acc.add(t, value)
        elif start is not None:
            acc.add(t, value)
            dev = thermal.deviation(value, low, high)
            if dev != 0 and abs(dev) > abs(thermal.deviation(peak, low, high)):
                peak = value
            cleared = low + margin <= value <= high - margin
            if cleared:
                if (t - start).total_seconds() >= min_seconds:
                    out.append(Excursion(start, t, peak, acc.total))
                start = None
                peak = None
                acc = thermal.SeverityAccumulator(low, high, max_gap_seconds)
    if start is not None and rows and (rows[-1][0] - start).total_seconds() >= min_seconds:
        out.append(Excursion(start, rows[-1][0], peak, acc.total))
    return out
