"""可估计样本段的构建：口径换算、范围变更分段与适用性依据。

本模块只负责把快照中的业务记录整理成可供时间序列方法使用的等距样本，
并保留每一步的来源与限制说明，不做任何统计推断。
"""
from __future__ import annotations

from collections import defaultdict

# 小样本下限：低于该长度的窗口不报告检验结论，只保留原始统计量。
MIN_SAMPLE = 30


def _latest(rows):
    picked = {}
    for row in rows:
        prior = picked.get(row["id"])
        if prior is None or (row["released_at"], row["revision"]) > (prior["released_at"], prior["revision"]):
            picked[row["id"]] = row
    return picked


def real_log_series(snapshot):
    """把各地收入换算为人民币实际收入并取自然对数。

    每个返回点都记录原始观测与换算记录编号，系数解释时可回溯：
    real_cny = 本币收入 * (人民币/本币) * 100 / 当地价格指数。
    取对数后，均衡方程中的斜率是实际收入之间的弹性（无量纲），
    截距是基期口径与计量单位共同决定的常数，不代表合作收益水平。
    """
    import math

    conversions = {(row["series"], row["month"]): row for row in snapshot["tables"]["conversions"]}
    grouped = defaultdict(list)
    for row in snapshot["tables"]["observations"]:
        rate = conversions.get((row["series"], row["month"]))
        if rate is None:
            continue
        if rate["cny_per_unit"] <= 0 or rate["price_index"] <= 0:
            raise ValueError("汇率与价格指数应为正数")
        real_cny = row["value"] * rate["cny_per_unit"] * 100 / rate["price_index"]
        if real_cny <= 0:
            raise ValueError("实际收入必须为正数才能取对数")
        grouped[row["series"]].append(
            {
                "series": row["series"],
                "month": row["month"],
                "scope": row["scope"],
                "log_real_cny": math.log(real_cny),
                "real_cny": real_cny,
                "source": row["id"],
                "conversion": rate["id"],
                "currency": rate["series"],
            }
        )
    return {name: sorted(rows, key=lambda item: item["month"]) for name, rows in grouped.items()}


def scope_breaks(snapshot):
    """读取不可比的范围变更；comparable=false 的区间不能与其他口径直接拼接。"""
    breaks = []
    for row in snapshot["tables"]["regimes"]:
        if not row.get("comparable", True):
            breaks.append(
                {
                    "series": row["series"],
                    "month_from": row["month_from"],
                    "month_to": row["month_to"],
                    "scope": row["scope"],
                    "source": row["id"],
                    "released_at": row["released_at"],
                }
            )
    return sorted(breaks, key=lambda item: (item["series"], item["month_from"]))


def build_segments(snapshot):
    """按范围变更把样本切成口径一致的估计段。

    规则：
    1. 任一序列存在不可比口径变更时，以变更起点为界切分，跨断点不拼接；
    2. 每段内所有序列必须月份连续且齐全，缺失则收缩到共同月份；
    3. 段长度低于 MIN_SAMPLE 标记为 evidence=insufficient，不支持检验判断。
    """
    series = real_log_series(snapshot)
    breaks = scope_breaks(snapshot)
    boundaries = sorted({item["month_from"] for item in breaks})
    cuts = [0] + boundaries + [None]

    segments = []
    for index in range(len(cuts) - 1):
        start, end = cuts[index], cuts[index + 1]
        panel = {}
        scopes_by_series = {}
        for name, rows in series.items():
            window = [row for row in rows if (start is None or row["month"] >= start) and (end is None or row["month"] < end)]
            if not window:
                continue
            panel[name] = window
            scopes_by_series[name] = sorted({row["scope"] for row in window})
        if not panel:
            continue
        common = set.intersection(*({row["month"] for row in rows} for rows in panel.values()))
        month_from, month_to = min(common), max(common)
        if month_to - month_from + 1 != len(common):
            # 存在缺口：收缩到连续公共区间，避免隐式插补。
            ordered = sorted(common)
            runs, run_start, previous = [], ordered[0], ordered[0]
            for month in ordered[1:]:
                if month != previous + 1:
                    runs.append((run_start, previous))
                    run_start = month
                previous = month
            runs.append((run_start, previous))
            run_start, run_end = max(runs, key=lambda pair: pair[1] - pair[0])
            month_from, month_to = run_start, run_end
        length = month_to - month_from + 1
        affected = sorted({item["series"] for item in breaks if start is not None and item["month_from"] == start})
        segments.append(
            {
                "segment_id": f"seg-{index + 1}",
                "month_from": month_from,
                "month_to": month_to,
                "n_months": length,
                "series": sorted(panel),
                "scopes_by_series": scopes_by_series,
                "break_series": affected,
                "evidence": "sufficient" if length >= MIN_SAMPLE else "insufficient",
                "basis": (
                    "口径一致且样本达到小样本下限"
                    if length >= MIN_SAMPLE
                    else "口径一致但样本不足，不支持协整秩与误差修正检验"
                ),
            }
        )
    return segments, breaks


def segment_panel(snapshot, segment):
    """取出某个估计段内对齐的对数实际收入面板及逐点来源。"""
    series = real_log_series(snapshot)
    lo, hi = segment["month_from"], segment["month_to"]
    panel, provenance = {}, []
    for name in segment["series"]:
        rows = [row for row in series[name] if lo <= row["month"] <= hi]
        panel[name] = [row["log_real_cny"] for row in rows]
        provenance.extend(
            {
                "series": name,
                "month": row["month"],
                "scope": row["scope"],
                "source": row["source"],
                "conversion": row["conversion"],
            }
            for row in rows
        )
    return panel, sorted(provenance, key=lambda item: (item["month"], item["series"]))
