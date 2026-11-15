from __future__ import annotations

from .analysis import METHOD, analyze, to_native
from .product import summarize


def publish_overview(client, as_of, request_key):
    snapshot = client.snapshot(as_of)
    policy = client.policy()
    artifact = {
        "as_of": as_of,
        "snapshot_id": snapshot["snapshot_id"],
        "method": "observed-overview-v1",
        "result": summarize(snapshot),
    }
    return client.publish(request_key, artifact, policy["revision"])


def build_equilibrium_artifact(snapshot, scenario=None, historical_backtest=None):
    """在本地资料边界内组装均衡分析成果，再由调用方通过客户端发布。

    历史检验需要完整资料束逐 vintage 重算，因此由本地调用方传入；
    HTTP 客户端本身只承担快照读取与不可变成果保存。
    """
    result = analyze(snapshot, scenario=scenario)
    if historical_backtest is not None:
        result["historical_backtest"] = historical_backtest
    return to_native(
        {
            "as_of": snapshot["as_of"],
            "snapshot_id": snapshot["snapshot_id"],
            "method": METHOD,
            "result": result,
            "items": sorted({row["series"] for row in snapshot["tables"]["observations"]}),
        }
    )


def publish_equilibrium_analysis(client, artifact, request_key):
    """发布已在本地组装好的均衡分析成果。"""
    policy = client.policy()
    return client.publish(request_key, artifact, policy["revision"])
