"""长期均衡分析的总装入口，产出可发布的结构化成果。"""

from __future__ import annotations

import numpy as np

from .data import visible_snapshot
from .equilibrium import cointegration_report, error_correction_model
from .scenarios import conditional_scenario, granger_report
from .segments import build_segments, segment_panel

METHOD = "equilibrium-analysis-v1"


def to_native(value):
    """递归把 numpy 标量/数组转为原生 JSON 类型，并拒绝非有限浮点。"""
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("分析结果出现非有限浮点")
        return number
    if isinstance(value, np.ndarray):
        return to_native(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("分析结果出现非有限浮点")
    return value


def analyze(snapshot, scenario=None):
    """对单个快照执行完整分析；scenario 为 None 时跳过条件情景。

    scenario 形如 {"source_series": "b", "shocks": [0.01]*6, "horizon": 6,
                   "band_alpha": 0.32}。
    """
    segments, breaks = build_segments(snapshot)
    segment_outputs = []
    for segment in segments:
        panel, provenance = segment_panel(snapshot, segment)
        coint_result = cointegration_report(panel)
        ecm = error_correction_model(panel, coint_result)
        granger = granger_report(panel)
        scenario_result = None
        if scenario is not None:
            scenario_result = conditional_scenario(
                panel,
                scenario["source_series"],
                scenario["shocks"],
                scenario.get("horizon", len(scenario["shocks"])),
                band_alpha=scenario.get("band_alpha", 0.32),
                origin_month=segment["month_to"],
            )
        segment_outputs.append(
            {
                "segment": segment,
                "unit_root": coint_result["unit_root"],
                "cointegration": {
                    key: value
                    for key, value in coint_result.items()
                    if key not in ("unit_root", "residual")
                },
                "error_correction": ecm,
                "granger_lead_lag": granger,
                "conditional_scenario": scenario_result,
                "provenance": provenance,
            }
        )
    return to_native(
        {
            "method": METHOD,
            "scope_breaks": breaks,
            "segments": segment_outputs,
            "interpretation": (
                "对数实际人民币收入之间的协整与误差修正分析；"
                "协整向量斜率为实际收入弹性，误差修正与条件响应均为统计关系而非因果效应"
            ),
        }
    )


def build_artifact(bundle, as_of, scenario=None, include_backtest=True):
    """在本地资料边界内组装完整成果，供工作流通过客户端发布。"""
    snapshot = visible_snapshot(bundle, as_of)
    result = analyze(snapshot, scenario=scenario)
    if include_backtest:
        from .backtest import backtest

        result["historical_backtest"] = to_native(
            backtest(
                bundle,
                scenario_source=scenario["source_series"] if scenario else None,
            )
        )
    return to_native(
        {
            "as_of": as_of,
            "snapshot_id": snapshot["snapshot_id"],
            "method": METHOD,
            "result": result,
            "items": sorted({row["series"] for row in snapshot["tables"]["observations"]}),
        }
    )
