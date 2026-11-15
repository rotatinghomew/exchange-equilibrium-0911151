"""条件路径情景与统计先后关系。

重要边界：这里的冲击分析是“在给定来源路径条件下的简化式预测”，
回答的是历史相关结构下路径如何联动，不识别外生冲击，
因此任何响应都不得称为因果效应，只称为条件响应或先后响应。
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import grangercausalitytests

from .segments import MIN_SAMPLE


def _finite(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("情景结果出现非有限值")
    return value


def differenced_panel(panel):
    return pd.DataFrame({name: np.diff(np.asarray(values, dtype=float)) for name, values in panel.items()})


def granger_report(panel, max_lag=2):
    """只报告统计上的先后可预测性，措辞固定为 lead-lag，不写因果。"""
    frame = differenced_panel(panel)
    if len(frame) <= max_lag + 3:
        return {"eligible": False, "tests": []}
    tests = []
    names = list(frame.columns)
    for target in names:
        for source in names:
            if source == target:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                output = grangercausalitytests(frame[[target, source]], maxlag=max_lag)
            for lag in range(1, max_lag + 1):
                p_value = output[lag][0]["ssr_ftest"][1]
                tests.append(
                    {
                        "target": target,
                        "leading_series": source,
                        "lag": lag,
                        "p_value": round(_finite(p_value), 4),
                        "leads_at_5pct": bool(p_value < 0.05),
                        "wording": "统计先后可预测性（lead-lag），不构成因果证据",
                    }
                )
    return {"eligible": True, "tests": tests}


def _fit_var(frame, max_lag=2):
    model = VAR(frame.values)
    return model.fit(maxlags=max_lag, ic="aic")


def conditional_scenario(panel, source_series, shocks, horizon, band_alpha=0.32, origin_month=None):
    """给定某个来源序列的差分路径，输出条件预测与不确定带。

    shocks：长度等于 horizon 的对数差分冲击序列（如 0.01 表示该月多增 1%）。
    band_alpha=0.32 对应约 68% 中心区间；同时给 95% 区间。
    适用期受估计段长度限制：训练样本低于下限则拒绝发布情景。
    """
    frame = differenced_panel(panel)
    names = list(frame.columns)
    if source_series not in names:
        raise ValueError("来源序列不在估计段内")
    if len(shocks) != horizon or horizon < 1:
        raise ValueError("冲击长度必须与预测期一致")
    if len(frame) < MIN_SAMPLE:
        return {"eligible": False, "reason": "估计段样本不足，不发布条件情景", "scenarios": []}

    fit = _fit_var(frame)
    history = frame.values
    source_index = names.index(source_series)

    def forecast_paths(initial, alpha):
        """返回中点与累计路径的不确定带。

        单步区间来自 forecast_interval；累计区间按各步预测误差方差
        随时间相加后开方得到，不能把单步上下界直接累加。
        """
        mid, lower, upper = fit.forecast_interval(initial, horizon, alpha=alpha)
        step_sigma = (upper - lower) / 2.0  # 对称正态带的单步标准差当量
        cumulative_sigma = np.cumsum(step_sigma**2, axis=0) ** 0.5
        cumulative_mid = np.cumsum(mid, axis=0)
        return mid, cumulative_mid, cumulative_mid - cumulative_sigma, cumulative_mid + cumulative_sigma

    baseline_initial = history[-fit.k_ar :]
    scenario_initial = baseline_initial.copy()
    # 条件通过改写来源序列的最近观测实现；随后各期冲击逐期注入并滚动。
    scenario_initial[:, source_index] = scenario_initial[:, source_index] + np.asarray(shocks[: fit.k_ar], dtype=float)

    base_mid, base_cum, base_lo68, base_hi68 = forecast_paths(baseline_initial, band_alpha)
    _, _, base_lo95, base_hi95 = forecast_paths(baseline_initial, 0.05)
    scen_mid, scen_cum, scen_lo68, scen_hi68 = forecast_paths(scenario_initial, band_alpha)
    _, _, scen_lo95, scen_hi95 = forecast_paths(scenario_initial, 0.05)

    # 若冲击期超过 VAR 记忆长度，逐期把剩余冲击加到来源序列的条件路径上，
    # 并同步平移累计中点；区间宽度由模型残差结构决定，不受冲击平移影响。
    if horizon > fit.k_ar:
        for step in range(fit.k_ar, horizon):
            scen_mid[step, source_index] += float(shocks[step])
            scen_cum[step, source_index] += float(shocks[step])

    start_month = (origin_month if origin_month is not None else len(frame)) + 1

    def package(mid, cum, low68, high68, low95, high95, conditioned):
        paths = {}
        for index, name in enumerate(names):
            paths[name] = {
                "month": [start_month + step for step in range(horizon)],
                "conditional_diff_mid": [round(_finite(v), 6) for v in mid[:, index]],
                "cumulative_log_change_mid": [round(_finite(v), 6) for v in cum[:, index]],
                "band_68_lower": [round(_finite(v), 6) for v in low68[:, index]],
                "band_68_upper": [round(_finite(v), 6) for v in high68[:, index]],
                "band_95_lower": [round(_finite(v), 6) for v in low95[:, index]],
                "band_95_upper": [round(_finite(v), 6) for v in high95[:, index]],
            }
        return {
            "conditioned": conditioned,
            "var_lag_order": int(fit.k_ar),
            "paths": paths,
        }

    baseline = package(base_mid, base_cum, base_lo68, base_hi68, base_lo95, base_hi95, False)
    scenario = package(scen_mid, scen_cum, scen_lo68, scen_hi68, scen_lo95, scen_hi95, True)
    delta = {}
    for index, name in enumerate(names):
        delta[name] = [round(_finite(v), 6) for v in (scen_cum[:, index] - base_cum[:, index])]
    return {
        "eligible": True,
        "source_series": source_series,
        "shocks_log_diff": [float(v) for v in shocks],
        "horizon_months": horizon,
        "applicable_from_month": start_month,
        "training_months": int(len(frame) + 1),
        "baseline": baseline,
        "conditional_scenario": scenario,
        "conditional_minus_baseline_cumulative": delta,
        "interpretation": (
            "给定来源序列路径下的简化式条件响应，仅反映历史联动结构；"
            "不识别外生冲击，不能解释为因果效应或合作收益"
        ),
    }
