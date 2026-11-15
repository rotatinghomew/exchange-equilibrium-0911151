import json
import math
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

from exchange_equilibrium.analysis import METHOD, analyze, build_artifact, to_native
from exchange_equilibrium.data import load_bundle, visible_snapshot
from exchange_equilibrium.equilibrium import cointegration_report, error_correction_model
from exchange_equilibrium.scenarios import conditional_scenario, granger_report
from exchange_equilibrium.segments import build_segments, segment_panel

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = {"source_series": "b", "shocks": [0.01] * 6, "horizon": 6}


class EquilibriumTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(ROOT / "fixtures/bundle.json")
        self.snapshot = visible_snapshot(self.bundle, "2026-01-01")
        self.segments, _ = build_segments(self.snapshot)

    def panels(self):
        return [segment_panel(self.snapshot, segment)[0] for segment in self.segments]

    def test_unit_root_and_cointegration_paths_diverge_on_conflict(self):
        pre_panel, post_panel = self.panels()
        pre = cointegration_report(pre_panel)
        post = cointegration_report(post_panel)
        self.assertTrue(pre["eligible"] and post["eligible"])
        # 前段：Johansen 检出秩而 EG 不显著，结论必须保留为冲突，不允许二选一。
        self.assertEqual(pre["rank_judgement"], "evidence_conflicted_judgement_withheld")
        self.assertTrue(pre["evidence_conflict"])
        self.assertIsNotNone(pre["johansen"]["rank_tests"])
        # 后段：两条路径都不支持协整。
        self.assertEqual(post["rank_judgement"], "no_cointegrating_relation_supported")
        self.assertFalse(post["evidence_conflict"])

    def test_cointegrating_vector_is_normalized_elasticity(self):
        pre_panel, _ = self.panels()
        result = cointegration_report(pre_panel)
        vector = result["engle_granger"]["cointegrating_vector"]
        self.assertEqual(result["normalized_on"], "a")
        self.assertSetEqual(set(vector), {"const", "b", "c"})
        # 弹性为有限实数；截距单独列示，不与弹性混同。
        for value in vector.values():
            self.assertTrue(math.isfinite(value))

    def test_error_correction_identifies_adjusting_series(self):
        pre_panel, _ = self.panels()
        result = cointegration_report(pre_panel)
        ecm = error_correction_model(pre_panel, result)
        self.assertIn("a", ecm["adjusting_series"])
        self.assertEqual(set(ecm["weak_exogenous"]), {"b", "c"})
        self.assertIn("不是因果效应", ecm["interpretation"])

    def test_short_sample_withholds_judgement(self):
        short = {name: values[:12] for name, values in self.panels()[0].items()}
        result = cointegration_report(short)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["rank_judgement"], "withheld")
        self.assertIsNone(result["johansen"])

    def test_granger_wording_never_claims_causality(self):
        report = granger_report(self.panels()[0])
        self.assertTrue(report["eligible"])
        self.assertTrue(all("不构成因果证据" in item["wording"] for item in report["tests"]))

    def test_conditional_scenario_has_bands_applicability_and_no_causal_claim(self):
        scenario = conditional_scenario(self.panels()[0], "b", [0.01] * 6, 6, origin_month=47)
        self.assertTrue(scenario["eligible"])
        self.assertEqual(scenario["applicable_from_month"], 48)
        path = scenario["conditional_scenario"]["paths"]["a"]
        self.assertEqual(len(path["band_95_lower"]), 6)
        for lower, upper in zip(path["band_95_lower"], path["band_95_upper"]):
            self.assertLess(lower, upper)
        self.assertIn("不能解释为因果效应", scenario["interpretation"])

    def test_scenario_rejects_short_training_sample(self):
        short = {name: values[:12] for name, values in self.panels()[0].items()}
        scenario = conditional_scenario(short, "b", [0.01] * 3, 3)
        self.assertFalse(scenario["eligible"])

    def test_full_artifact_is_json_safe_and_method_versioned(self):
        artifact = build_artifact(self.bundle, "2026-01-01", scenario=SCENARIO)
        self.assertEqual(artifact["method"], METHOD)
        encoded = json.dumps(artifact, ensure_ascii=False, allow_nan=False)
        self.assertIn("historical_backtest", encoded)
        # 情景发布必须带条件，不允许只有无条件预测值。
        decoded = json.loads(encoded)
        for segment in decoded["result"]["segments"]:
            scenario = segment["conditional_scenario"]
            self.assertIsNotNone(scenario)
            self.assertIn("baseline", scenario)
            self.assertIn("conditional_scenario", scenario)


if __name__ == "__main__":
    unittest.main()
