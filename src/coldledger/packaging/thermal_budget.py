"""Packaging thermal exposure budget.

The budget is the same time integral used by excursion detection and
deviation-case severity (see :mod:`coldledger.core.thermal`): worst-case
endpoint hold per interval, conservative rejection of sparse gaps.
"""
from dataclasses import dataclass
from datetime import datetime
from ..core import thermal


@dataclass(frozen=True)
class Sample:
    at: datetime
    celsius: float


def exposure(samples, low, high, max_gap_seconds=None):
    """Cumulative excursion severity*hours over a sampled trajectory."""
    return thermal.integrate_rows(samples, low, high, max_gap_seconds)


def exposure_segments(segments, low, high, max_gap_seconds=None):
    """Exposure over replayed segments (per-leg or per-compartment feeds).

    Segments are merged onto one timeline before integration, so boundary
    re-samples between segments are deduplicated and a segment replay does not
    double-count the shared instant.
    """
    rows = []
    for seg in segments:
        rows.extend(seg)
    return exposure(rows, low, high, max_gap_seconds)
