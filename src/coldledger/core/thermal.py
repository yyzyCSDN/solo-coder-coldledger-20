"""Shared time integration for temperature telemetry.

The sampling trajectory (excursion detection), the packaging exposure budget
and deviation-case severity all consume the *same* integral: severity(t)
integrated over time. Sparse data is handled conservatively — the worst
observed endpoint value is held across each interval, and gaps larger than
``max_gap_seconds`` are rejected rather than interpolated away.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ThermalPoint:
    """One temperature observation. Anything with ``at``/``celsius`` works too."""
    at: datetime
    celsius: float


def deviation(celsius: float, low: float, high: float) -> float:
    """Signed excursion magnitude: negative below band, positive above, 0 inside."""
    if celsius < low:
        return celsius - low
    if celsius > high:
        return celsius - high
    return 0.0


def severity(celsius: float, low: float, high: float) -> float:
    """Instantaneous excursion severity; cold/hot sides share one convention."""
    d = deviation(celsius, low, high)
    if d < 0:
        return (-d) ** 1.35
    if d > 0:
        return d ** 1.55
    return 0.0


def normalize_rows(rows, max_gap_seconds: float | None = None):
    """Sort/UTC-normalize points and reconcile repeated samples.

    Cross-segment replay and duplicate samples collapse: equal timestamps with
    equal values are deduplicated (replay is idempotent), equal timestamps with
    conflicting values raise ``ValueError``. Sparse trajectories whose adjacent
    gap exceeds ``max_gap_seconds`` are conservatively rejected.
    """
    points = sorted(
        ((r.at.astimezone(timezone.utc), float(r.celsius)) for r in rows),
        key=lambda x: x[0],
    )
    out: list[tuple[datetime, float]] = []
    last_t = None
    for t, v in points:
        if last_t is not None and t == last_t:
            if v != out[-1][1]:
                raise ValueError("conflicting samples at duplicate time")
            continue  # identical re-sample / replayed sample: count once
        out.append((t, v))
        last_t = t
    if max_gap_seconds is not None:
        for (a, _), (b, _) in zip(out, out[1:]):
            if (b - a).total_seconds() > max_gap_seconds:
                raise ValueError("sample gap exceeds defensible max gap")
    return out


def integrate_rows(rows, low: float, high: float, max_gap_seconds: float | None = None) -> float:
    """Integrate excursion severity over time with worst-case endpoint hold.

    Returns severity*hours. The severity on each interval ``[t_i, t_i+1]`` is
    the maximum severity of its two endpoints, so a short, severe peak at any
    sampled instant is fully weighted in both adjacent intervals instead of
    being missed by a left-endpoint rectangle rule.
    """
    pts = normalize_rows(rows, max_gap_seconds)
    return _integrate_normalized(pts, low, high)


def integrate_span(rows, start, end, low, high, max_gap_seconds: float | None = None) -> float:
    """Severity*hours over ``[start, end]``, clipping the trajectory to the span."""
    pts = normalize_rows(rows, max_gap_seconds)
    s, e = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if e <= s:
        raise ValueError("non-positive span")
    inner = [(t, v) for (t, v) in pts if s <= t <= e]
    if not inner:
        raise ValueError("no samples within span")
    lo_t, lo_v = inner[0]
    hi_t, hi_v = inner[-1]
    if lo_t > s:
        before = [v for (t, v) in pts if t < s]
        if not before:
            raise ValueError("span starts before first sample")
        inner.insert(0, (s, before[-1]))  # hold last observed value into span
    if hi_t < e:
        after = [v for (t, v) in pts if t > e]
        if not after:
            raise ValueError("span ends after last sample")
        inner.append((e, after[0]))  # hold first observed value beyond span
    return _integrate_normalized(inner, low, high)


def _integrate_normalized(pts, low, high) -> float:
    score = 0.0
    for (ta, va), (tb, vb) in zip(pts, pts[1:]):
        dt = (tb - ta).total_seconds()
        if dt <= 0:
            raise ValueError("duplicate time")
        sev = max(severity(va, low, high), severity(vb, low, high))
        score += sev * dt / 3600.0
    return score


class SeverityAccumulator:
    """Streaming form of :func:`integrate_rows` for single-pass consumers."""

    def __init__(self, low: float, high: float, max_gap_seconds: float | None = None):
        self.low, self.high, self.max_gap = low, high, max_gap_seconds
        self.total = 0.0
        self._last_t = None
        self._last_sev = None
        self._last_v = None

    def add(self, at, celsius: float) -> None:
        t = at.astimezone(timezone.utc)
        v = float(celsius)
        if self._last_t is not None:
            if t == self._last_t:
                if v != self._last_v:
                    raise ValueError("conflicting samples at duplicate time")
                return  # duplicate replay sample: no additional weight
            dt = (t - self._last_t).total_seconds()
            if self.max_gap is not None and dt > self.max_gap:
                raise ValueError("sample gap exceeds defensible max gap")
            sev = severity(v, self.low, self.high)
            self.total += max(self._last_sev, sev) * dt / 3600.0
        self._last_t, self._last_v, self._last_sev = t, v, severity(v, self.low, self.high)
