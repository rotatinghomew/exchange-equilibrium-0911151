"""长期均衡关系与短期偏离的统计判断。

方法立场：
- 先做单整阶数判断；阶数无法确定或样本不足的窗口搁置结论，
  存在 I(0) 序列时检验仍运行但显式注明关系可能由该平稳序列驱动；
- 协整同时报告 Engle-Granger 残差检验与 Johansen 系统检验，
  两者结论冲突时保留分歧，不人为选定一个方向；
- 协整向量以被解释变量系数归一，斜率解释为实际收入弹性；
- 误差修正项只描述向历史均衡位置的回复速度，不解释为因果效应。
"""
from __future__ import annotations

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    import statsmodels.api as sm

from .segments import MIN_SAMPLE


def _finite(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("统计结果出现非有限值")
    return value


def unit_root_report(panel):
    """对每个序列的水平值与一阶差分做 ADF，返回单整阶数判断。"""
    report = {}
    for name, values in panel.items():
        values = np.asarray(values, dtype=float)
        if len(values) < MIN_SAMPLE:
            report[name] = {
                "integration_order": "undetermined",
                "adf_level_p": None,
                "adf_diff_p": None,
                "lag_level": None,
                "lag_diff": None,
                "note": "样本短于 ADF 回归可用下限，单整阶数搁置",
            }
            continue
        level = adfuller(values, regression="c", autolag="aic")
        differenced = adfuller(np.diff(values), regression="c", autolag="aic")
        level_stationary = level[1] < 0.05
        diff_stationary = differenced[1] < 0.05
        if not level_stationary and diff_stationary:
            order = "I(1)"
        elif level_stationary:
            order = "I(0)"
        else:
            order = "undetermined"
        report[name] = {
            "integration_order": order,
            "adf_level_p": round(_finite(level[1]), 4),
            "adf_diff_p": round(_finite(differenced[1]), 4),
            "lag_level": int(level[2]),
            "lag_diff": int(differenced[2]),
        }
    return report


def _engle_granger(panel, target, regressors):
    y = np.asarray(panel[target], dtype=float)
    design = sm.add_constant(np.column_stack([np.asarray(panel[name], dtype=float) for name in regressors]))
    fit = sm.OLS(y, design).fit()
    residual = fit.resid
    stat, pvalue, _ = coint(y, design[:, 1:], trend="c", autolag="aic")
    params = {"const": _finite(fit.params[0])}
    params.update({name: _finite(coef) for name, coef in zip(regressors, fit.params[1:])})
    return {
        "test": "engle_granger",
        "normalized_on": target,
        "statistic": round(_finite(stat), 4),
        "p_value": round(_finite(pvalue), 4),
        "cointegrating_vector": {name: round(value, 4) for name, value in params.items()},
        "residual_adf_p": round(_finite(adfuller(residual, regression="c", autolag="aic")[1]), 4),
        "r_squared": round(_finite(fit.rsquared), 4),
        "residual": residual.tolist(),
    }


def _johansen(panel, names, k_ar_diff=1):
    data = np.column_stack([np.asarray(panel[name], dtype=float) for name in names])
    result = coint_johansen(data, det_order=0, k_ar_diff=k_ar_diff)
    ranks = []
    for rank in range(len(names)):
        ranks.append(
            {
                "null_rank": rank,
                "trace_statistic": round(_finite(result.lr1[rank]), 4),
                "trace_critical_90": round(_finite(result.cvt[rank, 0]), 4),
                "trace_critical_95": round(_finite(result.cvt[rank, 1]), 4),
                "max_eig_statistic": round(_finite(result.lr2[rank]), 4),
                "max_eig_critical_95": round(_finite(result.cvm[rank, 1]), 4),
            }
        )
    # 第一列向量归一到第一个序列，便于与 Engle-Granger 口径对照。
    vectors = []
    for column in range(result.evec.shape[1]):
        vec = result.evec[:, column]
        normalized = vec / vec[0]
        vectors.append({name: round(_finite(value), 4) for name, value in zip(names, normalized)})
    trace_rank = sum(stat > crit for stat, crit in zip(result.lr1, result.cvt[:, 1]))
    maxeig_rank = sum(stat > crit for stat, crit in zip(result.lr2, result.cvm[:, 1]))
    return {
        "test": "johansen",
        "lag_order": k_ar_diff,
        "rank_tests": ranks,
        "rank_by_trace_95": int(trace_rank),
        "rank_by_max_eig_95": int(maxeig_rank),
        "normalized_eigenvectors": vectors,
        "small_sample_note": (
            "Johansen 临界值为渐近分布；本资料窗口仅 36–48 个月度观测，"
            "统计量贴近临界值时秩判断不稳定，应与 Engle-Granger 及逐 vintage 结果并读"
            if len(data) < 60
            else None
        ),
    }


def cointegration_report(panel):
    """双路检验并显式记录证据是否一致；样本不足时只给原始统计量。"""
    names = sorted(panel)
    n = len(panel[names[0]])
    unit = unit_root_report(panel)
    if n < MIN_SAMPLE or any(item["integration_order"] == "undetermined" for item in unit.values()):
        return {
            "eligible": False,
            "reason": "样本不足或存在单整阶数无法确定的序列",
            "unit_root": unit,
            "engle_granger": None,
            "johansen": None,
            "rank_judgement": "withheld",
            "evidence_conflict": None,
        }

    target = names[0]
    regressors = names[1:]
    level_stationary = [name for name, item in unit.items() if item["integration_order"] == "I(0)"]

    eg = _engle_granger(panel, target, regressors)
    joh = _johansen(panel, names)

    eg_supports = eg["p_value"] < 0.05
    johansen_supports = joh["rank_by_trace_95"] >= 1 or joh["rank_by_max_eig_95"] >= 1
    if eg_supports and johansen_supports:
        judgement, conflict = "one_or_more_cointegrating_relations_supported", False
    elif not eg_supports and not johansen_supports:
        judgement, conflict = "no_cointegrating_relation_supported", False
    else:
        judgement, conflict = "evidence_conflicted_judgement_withheld", True
    return {
        "eligible": True,
        "unit_root": unit,
        "level_stationary_series": level_stationary,
        "level_stationary_note": (
            "存在本身平稳的序列；协整检验仍可运行，但所谓均衡关系可能仅由 I(0) 序列驱动，需结合误差修正方程解读"
            if level_stationary
            else None
        ),
        "engle_granger": {key: value for key, value in eg.items() if key != "residual"},
        "johansen": joh,
        "rank_judgement": judgement,
        "evidence_conflict": conflict,
        "residual": eg["residual"],
        "normalized_on": target,
    }


def error_correction_model(panel, coint_result):
    """估计每个序列的短期误差修正方程并判断弱外生性。

    方程：Δy_t = c + γ·ect_{t-1} + 滞后一阶差分 + ε_t。
    γ 显著为负表示该序列承担向均衡回复的调整；
    γ 不显著表示该序列对均衡偏离弱外生，不代表它“推动”了其他序列。
    """
    if not coint_result.get("eligible"):
        return {"eligible": False, "equations": {}, "weak_exogenous": [], "adjusting_series": []}
    names = sorted(panel)
    ect = np.asarray(coint_result["residual"], dtype=float)
    diffs = {name: np.diff(np.asarray(panel[name], dtype=float)) for name in names}
    equations = {}
    adjusting, weak = [], []
    for target in names:
        others = [name for name in names if name != target]
        columns = [ect[1:-1]]
        labels = ["ect_lag1"]
        for name in names:
            columns.append(diffs[name][:-1])
            labels.append(f"d{name}_lag1")
        design = sm.add_constant(np.column_stack(columns))
        fit = sm.OLS(diffs[target][1:], design).fit()
        gamma_p = float(fit.pvalues[1])
        coefficients = {label: round(_finite(value), 4) for label, value in zip(labels, fit.params[1:])}
        equations[target] = {
            "gamma_ect_lag1": coefficients["ect_lag1"],
            "gamma_p_value": round(_finite(gamma_p), 4),
            "short_run_coefficients": {key: value for key, value in coefficients.items() if key != "ect_lag1"},
            "r_squared": round(_finite(fit.rsquared), 4),
            "adjusts_to_disequilibrium": bool(gamma_p < 0.05 and fit.params[1] < 0),
        }
        if equations[target]["adjusts_to_disequilibrium"]:
            adjusting.append(target)
        else:
            weak.append(target)
    rho = sm.OLS(ect[1:], sm.add_constant(ect[:-1])).fit().params[1]
    half_life = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else None
    return {
        "eligible": True,
        "equations": equations,
        "adjusting_series": adjusting,
        "weak_exogenous": weak,
        "ect_half_life_months": round(_finite(half_life), 2) if half_life is not None else None,
        "interpretation": "γ 是对历史均衡偏离的统计回复速度，不是因果效应",
    }
