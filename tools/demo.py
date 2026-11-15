from pathlib import Path
import json
import sys
import tempfile
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exchange_equilibrium.analysis import METHOD
from exchange_equilibrium.backtest import backtest
from exchange_equilibrium.client import ProductClient
from exchange_equilibrium.data import load_bundle
from exchange_equilibrium.service import running_service
from exchange_equilibrium.store import ArtifactStore
from exchange_equilibrium.workflow import build_equilibrium_artifact, publish_equilibrium_analysis, publish_overview

SCENARIO = {"source_series": "b", "shocks": [0.01] * 6, "horizon": 6, "band_alpha": 0.32}


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root / "fixtures/bundle.json")
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(str(Path(directory) / "state.sqlite3"), bundle.get("policy"))
        try:
            with running_service(bundle, store) as base:
                client = ProductClient(base)
                published = publish_overview(client, "2026-01-01", "overview-demo")
                restored = client.artifact(published["artifact_id"])
                assert restored == published["artifact"]

                snapshot = client.snapshot("2026-01-01")
                history = backtest(bundle, scenario_source=SCENARIO["source_series"])
                artifact = build_equilibrium_artifact(snapshot, scenario=SCENARIO, historical_backtest=history)
                assert artifact["method"] == METHOD
                receipt = publish_equilibrium_analysis(client, artifact, "equilibrium-demo")
                fetched = client.artifact(receipt["artifact_id"])
                assert fetched == artifact

                summary = {
                    "产品": bundle["product"],
                    "概览成果编号": published["artifact_id"],
                    "均衡成果编号": receipt["artifact_id"],
                    "方法": fetched["method"],
                    "估计段": [
                        {
                            "区间": f"{item['segment']['month_from']}-{item['segment']['month_to']}",
                            "口径": item["segment"]["scopes_by_series"],
                            "协整判断": item["cointegration"]["rank_judgement"],
                            "证据冲突": item["cointegration"]["evidence_conflict"],
                            "误差修正序列": item["error_correction"].get("adjusting_series"),
                        }
                        for item in fetched["result"]["segments"]
                    ],
                    "历史检验vintage数": fetched["result"]["historical_backtest"]["vintages_evaluated"],
                    "保留分歧的vintage数": len(fetched["result"]["historical_backtest"]["conflicted_vintages"]),
                    "情景说明": fetched["result"]["segments"][0]["conditional_scenario"]["interpretation"],
                }
                print(json.dumps(summary, ensure_ascii=False, indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    main()
