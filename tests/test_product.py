from copy import deepcopy
from pathlib import Path
import json
import unittest

from exchange_equilibrium.data import load_bundle, visible_snapshot
from exchange_equilibrium.product import *


class ProductTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.snapshot = visible_snapshot(load_bundle(root / "fixtures/bundle.json"), "2026-01-01")

    def test_overview_is_finite_and_preserves_sources(self):
        original = deepcopy(self.snapshot)
        result = summarize(self.snapshot)
        self.assertTrue(result)
        self.assertIn("interpretation", result)
        json.dumps(result, allow_nan=False)
        self.assertEqual(self.snapshot, original)

    def test_real_currency_conversion_matches_source(self):
        rows = comparable_values(self.snapshot)
        row = rows[0]
        source = next(value for value in self.snapshot["tables"]["observations"] if value["id"] == row["source"])
        rate = next(value for value in self.snapshot["tables"]["conversions"] if value["id"] == row["conversion"])
        self.assertAlmostEqual(row["real_cny"],source["value"]*rate["cny_per_unit"]*100/rate["price_index"])

    def test_nonpositive_exchange_rate_rejected(self):
        self.snapshot["tables"]["conversions"][0]["cny_per_unit"] = 0
        with self.assertRaises(ValueError):
            comparable_values(self.snapshot)
