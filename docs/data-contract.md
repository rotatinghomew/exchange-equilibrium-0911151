# 业务资料契约

observations.month 为等距结算月份序号，公开日期使用压缩的本地资料日历，并不等于真实月末。series.currency 与 price_base 描述本地货币和指数基期。conversions 的 cny_per_unit 是一单位本币可换人民币数，price_index 为当地价格指数，以 100 为基期。regimes 的范围变更在指定月区间生效；comparable=false 表示不能直接拼成同口径长期序列。

## 资料表

| 表名 | 原始记录数 | 字段 |
| --- | ---: | --- |
| series | 3 | currency, id, price_base, released_at, revision |
| observations | 273 | id, month, released_at, revision, scope, series, value |
| conversions | 252 | cny_per_unit, id, month, price_index, released_at, revision, series |
| regimes | 1 | comparable, id, month_from, month_to, released_at, revision, scope, series |

记录均为本地合成资料，只表达业务结构，不是公开报道所述人群或机构的实测资料。日期是本地资料日历；所有版本以资料服务的 as_of 为准。字段值和自然缺失保留原义。金额、评分、频率和权重不可跨表不加判断地直接合计。
