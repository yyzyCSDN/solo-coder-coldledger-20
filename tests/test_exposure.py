import unittest
from datetime import datetime, timezone, timedelta

from coldledger.core.timeseries import (
    ConflictingSampleError,
    Sample,
    SegmentAccumulator,
    SparseDataError,
    band_severity,
    integrate,
    normalize,
)
from coldledger.claims.excursions import detect, severity_from_exposure
from coldledger.claims.excursions import Sample as ClaimSample
from coldledger.operations.deviation_board import open_from_excursion
from coldledger.packaging.thermal_budget import exposure, Sample as BudgetSample
from coldledger.telemetry.trajectory import trajectory_exposure, trajectory_accumulator
from coldledger.telemetry.watermark import Reading

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
LOW, HIGH = 2.0, 8.0


def at(**kwargs):
    return T0 + timedelta(**kwargs)


def budget_series(pairs):
    return [BudgetSample(at(minutes=m), v) for m, v in pairs]


class KernelTests(unittest.TestCase):
    def test_trapezoidal_integral(self):
        # linear ramp 0 -> 10 over one hour, identity severity: average 5 * 3600s
        rows = [Sample(T0, 0.0), Sample(at(hours=1), 10.0)]
        self.assertAlmostEqual(integrate(rows, lambda v: v), 5.0 * 3600.0)

    def test_empty_and_single_sample_integrate_to_zero(self):
        self.assertEqual(integrate([], lambda v: 1.0), 0.0)
        self.assertEqual(integrate([Sample(T0, 3.0)], lambda v: 1.0), 0.0)

    def test_exact_duplicates_are_idempotent(self):
        once = normalize([Sample(T0, 1.0), Sample(at(minutes=1), 2.0)])
        twice = normalize([Sample(T0, 1.0), Sample(T0, 1.0), Sample(at(minutes=1), 2.0)])
        self.assertEqual(once, twice)

    def test_conflicting_duplicates_rejected(self):
        with self.assertRaises(ConflictingSampleError):
            normalize([Sample(T0, 1.0), Sample(T0, 2.0)])
        self.assertTrue(issubclass(ConflictingSampleError, ValueError))

    def test_sparse_gap_rejected_beyond_budget(self):
        rows = [Sample(T0, 1.0), Sample(at(hours=2), 2.0)]
        with self.assertRaises(SparseDataError):
            normalize(rows, max_gap_seconds=3600)
        self.assertTrue(issubclass(SparseDataError, ValueError))
        # within budget: interpolated, not rejected
        self.assertEqual(len(normalize(rows, max_gap_seconds=7200)), 2)

    def test_non_finite_and_naive_rejected(self):
        with self.assertRaises(ValueError):
            normalize([Sample(T0, float("nan"))])
        with self.assertRaises(ValueError):
            normalize([Sample(datetime(2026, 1, 1), 1.0)])


class BudgetTests(unittest.TestCase):
    def test_terminal_spike_no_longer_invisible(self):
        # left-endpoint rule used to score this 0.0
        series = budget_series([(0, 5.0), (60, 5.0), (120, 40.0)])
        self.assertGreater(exposure(series, LOW, HIGH), 0.0)

    def test_trapezoid_weights_both_endpoints(self):
        series = budget_series([(0, 5.0), (60, 40.0)])
        sev = band_severity(LOW, HIGH)(40.0)
        self.assertAlmostEqual(exposure(series, LOW, HIGH), sev / 2.0 * 1.0)  # 1 hour ramp

    def test_duplicate_samples_do_not_change_exposure(self):
        plain = budget_series([(0, 5.0), (60, 20.0), (120, 5.0)])
        with_dupes = plain + [BudgetSample(at(minutes=60), 20.0)]
        self.assertAlmostEqual(exposure(plain, LOW, HIGH), exposure(with_dupes, LOW, HIGH))

    def test_sparse_gap_rejected(self):
        series = budget_series([(0, 20.0), (600, 20.0)])
        with self.assertRaises(SparseDataError):
            exposure(series, LOW, HIGH, max_gap_seconds=3600)


class ExcursionTests(unittest.TestCase):
    def test_brief_severe_peak_produces_case(self):
        # 35C for four minutes: the old duration debounce dropped this silently
        series = [ClaimSample(at(minutes=0), 5.0), ClaimSample(at(minutes=6), 35.0), ClaimSample(at(minutes=10), 5.0)]
        (exc,) = detect(series, LOW, HIGH, margin=0.5)
        self.assertEqual(exc.peak, 35.0)
        self.assertGreater(exc.exposure, 0.0)
        self.assertGreaterEqual(severity_from_exposure(exc.exposure), 1)

    def test_excursion_exposures_sum_to_budget(self):
        pairs = [(0, 5.0), (10, 12.0), (20, 5.0), (30, 5.0), (40, 1.0), (50, 5.0), (60, 30.0)]
        claims = [ClaimSample(at(minutes=m), v) for m, v in pairs]
        total = exposure(budget_series(pairs), LOW, HIGH)
        self.assertAlmostEqual(sum(e.exposure for e in detect(claims, LOW, HIGH, margin=0.5)), total)

    def test_unclosed_excursion_uses_last_sample(self):
        claims = [ClaimSample(at(minutes=0), 5.0), ClaimSample(at(minutes=10), 20.0)]
        (exc,) = detect(claims, LOW, HIGH, margin=0.5)
        self.assertEqual(exc.end, at(minutes=10))
        self.assertGreater(exc.exposure, 0.0)

    def test_severity_mapping(self):
        self.assertEqual(severity_from_exposure(0.0), 1)
        self.assertEqual(severity_from_exposure(0.5), 2)
        self.assertEqual(severity_from_exposure(100.0), 5)
        with self.assertRaises(ValueError):
            severity_from_exposure(1.0, thresholds=(2.0, 2.0))

    def test_deviation_case_inherits_integral_severity(self):
        claims = [ClaimSample(at(minutes=0), 5.0), ClaimSample(at(minutes=30), 30.0), ClaimSample(at(minutes=60), 5.0)]
        (exc,) = detect(claims, LOW, HIGH, margin=0.5)
        case = open_from_excursion("DEV-1", exc, owner="qa", opened_at=at(minutes=61))
        self.assertEqual(case.severity, severity_from_exposure(exc.exposure))


class TrajectoryTests(unittest.TestCase):
    def test_trajectory_matches_budget(self):
        pairs = [(0, 5.0), (15, 25.0), (30, 5.0), (45, 1.0), (60, 5.0)]
        readings = [Reading("dev-1", at(minutes=m), v, i) for i, (m, v) in enumerate(pairs)]
        self.assertAlmostEqual(
            trajectory_exposure(readings, LOW, HIGH),
            exposure(budget_series(pairs), LOW, HIGH),
        )


class SegmentReplayTests(unittest.TestCase):
    def test_segments_stitch_to_one_pass_total(self):
        pairs = [(0, 5.0), (10, 20.0), (20, 5.0), (30, 1.0), (40, 5.0)]
        seg_a = [Sample(at(minutes=m), v) for m, v in pairs[:3]]
        seg_b = [Sample(at(minutes=m), v) for m, v in pairs[2:]]  # shared boundary sample
        acc = SegmentAccumulator(band_severity(LOW, HIGH))
        acc.add(seg_a)
        stitched = acc.add(seg_b)
        one_pass = integrate([Sample(at(minutes=m), v) for m, v in pairs], band_severity(LOW, HIGH))
        self.assertAlmostEqual(stitched, one_pass)

    def test_replay_is_idempotent(self):
        seg = [Sample(at(minutes=m), v) for m, v in [(0, 5.0), (10, 20.0), (20, 5.0)]]
        acc = SegmentAccumulator(band_severity(LOW, HIGH))
        first = acc.add(seg)
        self.assertEqual(acc.add(seg), first)  # cross-segment replay
        self.assertEqual(acc.add(list(reversed(seg))), first)  # out-of-order replay

    def test_conflicting_replay_rejected_and_state_kept(self):
        seg = [Sample(at(minutes=m), v) for m, v in [(0, 5.0), (10, 20.0)]]
        acc = SegmentAccumulator(band_severity(LOW, HIGH))
        total = acc.add(seg)
        with self.assertRaises(ConflictingSampleError):
            acc.add([Sample(at(minutes=10), 21.0)])
        self.assertEqual(acc.total, total)

    def test_late_sample_inserts_into_history(self):
        acc = SegmentAccumulator(band_severity(LOW, HIGH))
        acc.add([Sample(at(minutes=0), 5.0), Sample(at(minutes=20), 5.0)])
        base = acc.total
        acc.add([Sample(at(minutes=10), 30.0)])  # late-arriving spike
        self.assertGreater(acc.total, base)

    def test_telemetry_accumulator_matches_budget(self):
        pairs = [(0, 5.0), (10, 25.0), (20, 5.0)]
        acc = trajectory_accumulator(LOW, HIGH)
        acc.add([Sample(at(minutes=m), v) for m, v in pairs])
        self.assertAlmostEqual(acc.total / 3600.0, exposure(budget_series(pairs), LOW, HIGH))


if __name__ == "__main__":
    unittest.main()
