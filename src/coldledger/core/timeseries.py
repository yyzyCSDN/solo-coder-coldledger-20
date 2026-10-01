"""Shared time integral for sampled series.

Sampling trajectories, exposure budgets and deviation cases must all
integrate exposure the same way, otherwise a peak that one path counts
is invisible to another. This module is the single implementation.

Decisions baked in here:

* Integration is trapezoidal over linearly interpolated samples. Pure
  endpoint (step) rules give the first/last sample of a series zero or
  double weight, so a brief severe spike at a series boundary used to
  vanish from the budget.
* Sparse data: gaps up to ``max_gap_seconds`` are interpolated; a wider
  gap is rejected with :class:`SparseDataError` instead of fabricating
  exposure across an unobserved window. ``max_gap_seconds=None`` opts in
  to unbounded interpolation explicitly.
* Duplicate sampling: an exact duplicate (same instant, same value) is
  dropped so replays are idempotent; the same instant with a different
  value is a genuine conflict and raises :class:`ConflictingSampleError`.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Callable, Iterable
from .clock import utc


@dataclass(frozen=True)
class Sample:
    at: datetime
    value: float


class SparseDataError(ValueError):
    """A gap between consecutive samples exceeds the interpolation budget."""


class ConflictingSampleError(ValueError):
    """Two samples share an instant but disagree on the value."""


def normalize(samples: Iterable[Sample], *, max_gap_seconds: float | None = None) -> tuple[Sample, ...]:
    """Sort, deduplicate and gap-check a sampled series.

    Idempotent: normalizing an already normalized series changes nothing,
    which is what makes cross-segment replay safe to re-feed.
    """
    if max_gap_seconds is not None and max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    rows = sorted((Sample(utc(s.at), float(s.value)) for s in samples), key=lambda s: s.at)
    out: list[Sample] = []
    for s in rows:
        if not isfinite(s.value):
            raise ValueError("non-finite sample value")
        if out and s.at == out[-1].at:
            if s.value == out[-1].value:
                continue  # exact replay of an already seen sample
            raise ConflictingSampleError(f"conflicting values at {s.at.isoformat()}")
        out.append(s)
    if max_gap_seconds is not None:
        for a, b in zip(out, out[1:]):
            gap = (b.at - a.at).total_seconds()
            if gap > max_gap_seconds:
                raise SparseDataError(
                    f"gap of {gap:.0f}s between {a.at.isoformat()} and {b.at.isoformat()} "
                    f"exceeds interpolation budget {max_gap_seconds:.0f}s"
                )
    return tuple(out)


def band_severity(low: float, high: float) -> Callable[[float], float]:
    """The shared exposure law: 0 inside [low, high], superlinear outside."""
    if not low < high:
        raise ValueError("low must be below high")

    def severity(celsius: float) -> float:
        if celsius < low:
            return (low - celsius) ** 1.35
        if celsius > high:
            return (celsius - high) ** 1.55
        return 0.0

    return severity


def _integrate_normalized(rows: tuple[Sample, ...], severity: Callable[[float], float]) -> float:
    total = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = (b.at - a.at).total_seconds()
        total += (severity(a.value) + severity(b.value)) / 2.0 * dt
    return total


def integrate(
    samples: Iterable[Sample],
    severity: Callable[[float], float],
    *,
    max_gap_seconds: float | None = None,
) -> float:
    """Trapezoidal time integral of ``severity(value(t))`` in severity-seconds."""
    return _integrate_normalized(normalize(samples, max_gap_seconds=max_gap_seconds), severity)


class SegmentAccumulator:
    """Stitches consecutively produced segments into one running integral.

    The boundary interval between the last sample of one segment and the
    first sample of the next is integrated exactly once, so per-segment
    processing sums to the same total as one pass over the full series.
    Re-adding an identical segment (cross-segment replay) is a no-op;
    re-adding it with altered values raises ConflictingSampleError and
    leaves the accumulator untouched.
    """

    def __init__(self, severity: Callable[[float], float], *, max_gap_seconds: float | None = None):
        self._severity = severity
        self._max_gap_seconds = max_gap_seconds
        self._samples: tuple[Sample, ...] = ()
        self._total = 0.0

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._samples

    @property
    def total(self) -> float:
        """Running integral in severity-seconds."""
        return self._total

    def add(self, segment: Iterable[Sample]) -> float:
        merged = normalize((*self._samples, *segment), max_gap_seconds=self._max_gap_seconds)
        self._total = _integrate_normalized(merged, self._severity)
        self._samples = merged
        return self._total
