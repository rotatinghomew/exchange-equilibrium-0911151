import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

from exchange_equilibrium.data import load_bundle, visible_snapshot
from exchange_equilibrium.segments import MIN_SAMPLE, build_segments, real_log_series, segment_panel

ROOT = Path(__file__).resolve().parents[1]


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(ROOT / "fixtures/bundle.json")
        self.snapshot = visible_snapshot(self.bundle, "2026-01-01")

    def test_scope_break_splits_sample_and_keeps_scopes_apart(self):
        segments, breaks = build_segments(self.snapshot)
        self.assertEqual(len(breaks), 1)
        self.assertEqual(breaks[0]["series"], "c")
        self.assertEqual(len(segments), 2)
        first, second = segments
        self.assertEqual((first["month_from"], first["month_to"]), (0, 47))
        self.assertEqual((second["month_from"], second["month_to"]), (48, 83))
        self.assertEqual(first["scopes_by_series"]["c"], ["original"])
        self.assertEqual(second["scopes_by_series"]["c"], ["expanded"])
        self.assertTrue(all(segment["n_months"] >= MIN_SAMPLE for segment in segments))

    def test_log_series_preserves_provenance_and_formula(self):
        series = real_log_series(self.snapshot)
        row = series["b"][0]
        rate = next(item for item in self.snapshot["tables"]["conversions"] if item["id"] == row["conversion"])
        observed = next(item for item in self.snapshot["tables"]["observations"] if item["id"] == row["source"])
        import math

        expected = math.log(observed["value"] * rate["cny_per_unit"] * 100 / rate["price_index"])
        self.assertAlmostEqual(row["log_real_cny"], expected)

    def test_panel_is_aligned_and_continuous(self):
        segments, _ = build_segments(self.snapshot)
        for segment in segments:
            panel, provenance = segment_panel(self.snapshot, segment)
            lengths = {len(values) for values in panel.values()}
            self.assertEqual(lengths, {segment["n_months"]})
            months = sorted({row["month"] for row in provenance})
            self.assertEqual(months, list(range(segment["month_from"], segment["month_to"] + 1)))


if __name__ == "__main__":
    unittest.main()
