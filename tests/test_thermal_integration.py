import unittest
from datetime import datetime, timezone, timedelta

from coldledger.packaging.thermal_budget import Sample, exposure, exposure_segments
from coldledger.claims.excursions import detect
from coldledger.core import thermal
from coldledger.operations.exception_scoring import case_from_excursion, raw_score

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def S(minutes, celsius):
    return Sample(T0 + timedelta(minutes=minutes), celsius)


class ThermalIntegrationTests(unittest.TestCase):
    def test_short_severe_intermediate_peak_is_counted(self):
        # A 1-minute spike to 20C between in-band 5C samples must register.
        with_peak = [S(0, 5.0), S(59, 5.0), S(60, 20.0), S(61, 5.0), S(120, 5.0)]
        flat = [S(0, 5.0), S(59, 5.0), S(61, 5.0), S(120, 5.0)]
        self.assertGreater(exposure(with_peak, 2, 8), 0.0)
        self.assertEqual(exposure(flat, 2, 8), 0.0)

    def test_peak_at_final_sample_not_missed(self):
        # Left-endpoint rectangle rule scored this as exactly zero.
        rows = [S(0, 5.0), S(59, 5.0), S(60, 20.0)]
        self.assertGreater(exposure(rows, 2, 8), 0.0)

    def test_sparse_gap_is_conservatively_rejected(self):
        with self.assertRaises(ValueError):
            exposure([S(0, 5.0), S(120, 5.0)], 2, 8, max_gap_seconds=3600)

    def test_gap_within_limit_accepted(self):
        self.assertEqual(
            exposure([S(0, 5.0), S(60, 5.0)], 2, 8, max_gap_seconds=3600),
            0.0,
        )

    def test_duplicate_replayed_samples_are_idempotent(self):
        rows = [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        replayed = rows + [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        self.assertEqual(exposure(rows, 2, 8), exposure(replayed, 2, 8))

    def test_conflicting_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):
            exposure([S(0, 5.0), S(30, 20.0), S(30, 21.0), S(60, 5.0)], 2, 8)

    def test_cross_segment_replay_matches_single_timeline(self):
        seg1 = [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        seg2 = [S(60, 5.0), S(90, 20.0), S(120, 5.0)]
        merged = [S(0, 5.0), S(30, 20.0), S(60, 5.0), S(90, 20.0), S(120, 5.0)]
        self.assertEqual(exposure_segments([seg1, seg2], 2, 8), exposure(merged, 2, 8))

    def test_excursion_and_budget_share_one_integral(self):
        rows = [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        excs = detect(rows, 2, 8, 0.0, 1)
        self.assertEqual(len(excs), 1)
        window = thermal.integrate_span(rows, excs[0].start, excs[0].end, 2, 8)
        self.assertEqual(excs[0].severity_integral, window)

    def test_identical_excursions_get_consistent_case_severity(self):
        rows = [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        a = detect(rows, 2, 8, 0.0, 1)[0]
        b = detect(list(rows), 2, 8, 0.0, 1)[0]
        now = T0 + timedelta(hours=6)
        self.assertEqual(
            raw_score(case_from_excursion(a, "A", now)),
            raw_score(case_from_excursion(b, "B", now)),
        )

    def test_detect_reconciles_replayed_segments(self):
        seg1 = [S(0, 5.0), S(30, 20.0), S(60, 5.0)]
        seg2 = [S(60, 5.0), S(90, 20.0), S(120, 5.0)]
        merged = [S(0, 5.0), S(30, 20.0), S(60, 5.0), S(90, 20.0), S(120, 5.0)]
        self.assertEqual(
            [e.severity_integral for e in detect(seg1 + seg2, 2, 8, 0.0, 1)],
            [e.severity_integral for e in detect(merged, 2, 8, 0.0, 1)],
        )

    def test_cold_side_uses_same_integral(self):
        rows = [S(0, 5.0), S(30, -10.0), S(60, 5.0)]
        self.assertGreater(exposure(rows, 2, 8), 0.0)
        excs = detect(rows, 2, 8, 0.0, 1)
        self.assertEqual(len(excs), 1)
        self.assertEqual(excs[0].peak, -10.0)
        self.assertEqual(
            excs[0].severity_integral,
            thermal.integrate_span(rows, excs[0].start, excs[0].end, 2, 8),
        )

    def test_empty_and_single_sample(self):
        self.assertEqual(exposure([], 2, 8), 0.0)
        self.assertEqual(exposure([S(0, 5.0)], 2, 8), 0.0)


if __name__ == "__main__":
    unittest.main()
