"""Exposure over device sampling trajectories.

Device readings (e.g. telemetry.watermark.Reading, after clock-drift
correction) integrate through the same kernel as exposure budgets and
deviation cases, so a peak counted here is counted identically there.
"""
from __future__ import annotations
from typing import Iterable
from ..core.timeseries import Sample, SegmentAccumulator, band_severity, integrate


def trajectory_exposure(readings: Iterable, low: float, high: float, *, max_gap_seconds: float | None = None) -> float:
    """Exposure of a sampling trajectory in severity-hours."""
    points = [Sample(r.event_time, r.value) for r in readings]
    return integrate(points, band_severity(low, high), max_gap_seconds=max_gap_seconds) / 3600.0


def trajectory_accumulator(low: float, high: float, *, max_gap_seconds: float | None = None) -> SegmentAccumulator:
    """Running exposure across segments; replayed segments are idempotent."""
    return SegmentAccumulator(band_severity(low, high), max_gap_seconds=max_gap_seconds)
