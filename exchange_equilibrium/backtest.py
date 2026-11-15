"""发布当时资料（point-in-time）的历史检验。

每个历史时点只使用该 as_of 可见的版本，事后修订不进入检验；
对协整证据冲突或样本不足的区间，结论保留为分歧/搁置而不是强行判定。
"""
from __future__ import annotations

from .data import visible_snapshot
from .equilibrium import cointegration_report, error_correction_model
from .scenarios import conditional_scenario
from .segments import build_segments, segment_panel


def vintage_calendar(bundle):
    """生成 (as_of, 当时可见月份范围) 列表，每个最大月份只保留最早公开日。"""
    release_dates = sorted({row["released_at"] for row in bundle["tables"]["observations"]})
    calendar = []
    for as_of in release_dates:
        snapshot = visible_snapshot(bundle, as_of)
        months = [row["month"] for row in snapshot["tables"]["observations"]]
        calendar.append((as_of, min(months), max(months)))
    # 同一最大月份只保留最早可见日期，避免重复 vintages。
    deduped = {}
    for as_of, lo, hi in calendar:
        deduped.setdefault(hi, (as_of, lo))
    return [{"as_of": as_of, "month_from": lo, "month_to": hi} for hi, (as_of, lo) in sorted(deduped.items())]


def backtest(bundle, min_vintage_months=30, scenario_source=None, scenario_horizon=6):
    """逐 vintage 重跑分段与协整判断，记录结论随资料积累的变化。"""
    entries = []
    for point in vintage_calendar(bundle):
        if point["month_to"] - point["month_from"] + 1 < min_vintage_months:
            continue
        snapshot = visible_snapshot(bundle, point["as_of"])
        segments, breaks = build_segments(snapshot)
        segment_results = []
        for segment in segments:
            panel, _ = segment_panel(snapshot, segment)
            coint_result = cointegration_report(panel)
            ecm = error_correction_model(panel, coint_result)
            scenario = None
            if scenario_source is not None and coint_result.get("eligible"):
                shocks = [0.01] * scenario_horizon
                scenario = conditional_scenario(
                    panel,
                    scenario_source,
                    shocks,
                    scenario_horizon,
                    origin_month=segment["month_to"],
                )
            segment_results.append(
                {
                    "segment_id": segment["segment_id"],
                    "month_from": segment["month_from"],
                    "month_to": segment["month_to"],
                    "n_months": segment["n_months"],
                    "evidence": segment["evidence"],
                    "scopes_by_series": segment["scopes_by_series"],
                    "rank_judgement": coint_result["rank_judgement"],
                    "evidence_conflict": coint_result["evidence_conflict"],
                    "engle_granger_p": (coint_result["engle_granger"] or {}).get("p_value"),
                    "johansen_trace_rank": (coint_result["johansen"] or {}).get("rank_by_trace_95"),
                    "adjusting_series": ecm.get("adjusting_series", []),
                    "weak_exogenous": ecm.get("weak_exogenous", []),
                    "scenario_eligible": None if scenario is None else scenario["eligible"],
                }
            )
        entries.append(
            {
                "as_of": point["as_of"],
                "visible_month_from": point["month_from"],
                "visible_month_to": point["month_to"],
                "regime_breaks_visible": [
                    {"series": item["series"], "month_from": item["month_from"]} for item in breaks
                ],
                "segments": segment_results,
            }
        )

    conflicted = [
        entry["as_of"]
        for entry in entries
        if any(segment["evidence_conflict"] for segment in entry["segments"])
    ]
    withheld = [
        entry["as_of"]
        for entry in entries
        if any(segment["rank_judgement"] == "withheld" for segment in entry["segments"])
    ]
    return {
        "method": "point_in_time_vintage_backtest",
        "vintages_evaluated": len(entries),
        "conflicted_vintages": conflicted,
        "withheld_vintages": withheld,
        "entries": entries,
        "interpretation": "每个 vintage 仅使用当时可见资料；冲突区间保留分歧，不用事后修订覆盖历史判断",
    }
