"""Excursion detection for deviation cases.

Each detected excursion carries ``exposure``: the same trapezoidal time
integral the exposure budget uses (severity-hours, see
coldledger.core.timeseries). The integration window extends from the last
in-band sample before the breach to the sample that clears it, so the
ramp in and out of the band is counted and the exposures of all
excursions in a series sum exactly to the budget total for that series.

Case-worthiness is severity-driven: the old minimum-duration debounce
silently dropped brief but severe peaks, so it was removed. A sensor
blip now contributes negligible exposure instead of needing a duration
filter; threshold on ``severity_from_exposure`` downstream.
"""
from dataclasses import dataclass
from datetime import datetime
from ..core.timeseries import Sample as Point, band_severity, integrate, normalize

#: Default cut-points (severity-hours) mapping an excursion's exposure to
#: deviation severity 1..5. Reference defaults; callers may override.
DEFAULT_SEVERITY_THRESHOLDS = (0.5, 2.0, 8.0, 32.0)


@dataclass(frozen=True)
class Sample:
    at: datetime
    value: float


@dataclass(frozen=True)
class Excursion:
    start: datetime
    end: datetime
    peak: float
    exposure: float  # severity-hours, same unit as packaging.thermal_budget.exposure


def detect(samples, low, high, margin=0.0, *, max_gap_seconds=None):
    if margin < 0:
        raise ValueError("negative margin")
    severity = band_severity(low, high)
    rows = normalize((Point(s.at, s.value) for s in samples), max_gap_seconds=max_gap_seconds)
    out = []
    i = 0
    while i < len(rows):
        if not (rows[i].value < low or rows[i].value > high):
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and not (low + margin <= rows[j + 1].value <= high - margin):
            j += 1
        window = rows[i - 1 if i > 0 else i:j + 2 if j + 1 < len(rows) else j + 1]
        breach = rows[i:j + 1]
        out.append(Excursion(
            start=rows[i].at,
            end=rows[j + 1].at if j + 1 < len(rows) else rows[j].at,
            peak=max(breach, key=lambda s: severity(s.value)).value,
            exposure=integrate(window, severity) / 3600.0,
        ))
        i = j + 1
    return out


def severity_from_exposure(exposure, thresholds=DEFAULT_SEVERITY_THRESHOLDS):
    """Map an excursion's exposure (severity-hours) to deviation severity."""
    if exposure < 0:
        raise ValueError("negative exposure")
    tiers = tuple(thresholds)
    if any(t <= 0 for t in tiers) or any(b <= a for a, b in zip(tiers, tiers[1:])):
        raise ValueError("thresholds must be positive and strictly increasing")
    return 1 + sum(exposure >= t for t in tiers)
