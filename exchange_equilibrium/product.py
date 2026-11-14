from collections import defaultdict


def comparable_values(snapshot):
    conversion = {(row["series"],row["month"]): row for row in snapshot["tables"]["conversions"]}
    values = []
    for row in snapshot["tables"]["observations"]:
        rate = conversion.get((row["series"],row["month"]))
        if rate is None:
            continue
        if rate["cny_per_unit"] <= 0 or rate["price_index"] <= 0:
            raise ValueError("汇率与价格指数应为正数")
        values.append({"series": row["series"], "month": row["month"], "real_cny": row["value"]*rate["cny_per_unit"]*100/rate["price_index"], "scope": row["scope"], "source": row["id"], "conversion": rate["id"]})
    return sorted(values, key=lambda row: (row["series"],row["month"]))


def summarize(snapshot):
    grouped = defaultdict(list)
    for row in comparable_values(snapshot):
        grouped[row["series"]].append(row)
    return {"series": {name: {"observations": len(rows), "first_real_cny": rows[0]["real_cny"], "last_real_cny": rows[-1]["real_cny"], "scopes": sorted({row["scope"] for row in rows})} for name, rows in grouped.items()}, "interpretation": "仅做币值和价格口径核对，不将共同趋势解释为均衡或因果关系"}
