import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

from exchange_equilibrium.backtest import backtest, vintage_calendar
from exchange_equilibrium.data import load_bundle, visible_snapshot

ROOT = Path(__file__).resolve().parents[1]


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(ROOT / "fixtures/bundle.json")

    def test_vintages_only_use_data_visible_at_the_time(self):
        calendar = vintage_calendar(self.bundle)
        self.assertGreater(len(calendar), 20)
        for point in calendar:
            snapshot = visible_snapshot(self.bundle, point["as_of"])
            max_month = max(row["month"] for row in snapshot["tables"]["observations"])
            self.assertEqual(max_month, point["month_to"])
            self.assertTrue(all(row["released_at"] <= point["as_of"] for row in snapshot["tables"]["observations"]))

    def test_backtest_keeps_conflict_and_withheld_regions_instead_of_resolving(self):
        report = backtest(self.bundle, scenario_source="b")
        self.assertGreater(report["vintages_evaluated"], 20)
        # 范围变更公开前的 vintage 不应看到 regime 记录。
        early = next(entry for entry in report["entries"] if entry["as_of"] < "2024-05-29")
        self.assertEqual(early["regime_breaks_visible"], [])
        later = next(entry for entry in report["entries"] if entry["as_of"] >= "2024-05-29")
        self.assertTrue(any(item["series"] == "c" for item in later["regime_breaks_visible"]))
        # 必须存在保留分歧或搁置判断的区间，且被显式列出。
        self.assertTrue(report["conflicted_vintages"] or report["withheld_vintages"])
        judgements = {
            segment["rank_judgement"]
            for entry in report["entries"]
            for segment in entry["segments"]
        }
        self.assertIn("evidence_conflicted_judgement_withheld", judgements)

    def test_backtest_never_mutates_bundle(self):
        from exchange_equilibrium.data import fingerprint

        before = fingerprint(self.bundle)
        backtest(self.bundle)
        self.assertEqual(fingerprint(self.bundle), before)


if __name__ == "__main__":
    unittest.main()
