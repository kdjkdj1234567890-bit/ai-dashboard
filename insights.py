"""Rule-based auto insight engine.

외부 AI API 없이 동작하는 룰 기반 인사이트 생성 엔진.
입력 DataFrame에서 컬럼 타입을 자동 인식하고,
KPI / 차트 스펙 / 자연어 인사이트를 생성한다.
모든 수치는 실제 데이터에서 계산되며, 하드코딩된 값이 없다.
"""
import numpy as np
import pandas as pd

ANOMALY_Z_THRESHOLD = 2.0
MAX_CATEGORY_UNIQUE = 15

PALETTE = [
    "#4F7CFF", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6",
    "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#6366F1",
]


# ---------------------------------------------------------------- column type
def detect_columns(df: pd.DataFrame) -> dict:
    """각 컬럼을 'date' | 'numeric' | 'category' 로 분류."""
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            result[col] = "category"
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.8:
            result[col] = "numeric"
            continue
        try:
            dt = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", format="mixed")
        except Exception:
            dt = pd.Series([pd.NaT] * len(s))
        if dt.notna().mean() >= 0.8:
            result[col] = "date"
            continue
        result[col] = "category"
    return result


def pick_primary_numeric(df: pd.DataFrame, coltypes: dict) -> str | None:
    """대표 지표 컬럼 선택: 매출/금액 계열 이름 우선, 없으면 첫 숫자 컬럼."""
    numerics = [c for c, t in coltypes.items() if t == "numeric"]
    if not numerics:
        return None
    keywords = ["매출", "금액", "수익", "매출액", "판매액", "sales", "revenue", "amount", "value"]
    for c in numerics:
        name = str(c).lower()
        if any(k in name for k in keywords):
            return c
    return numerics[0]


def pick_date_column(df: pd.DataFrame, coltypes: dict) -> str | None:
    dates = [c for c, t in coltypes.items() if t == "date"]
    return dates[0] if dates else None


# ---------------------------------------------------------------- period series
def period_series(df: pd.DataFrame, date_col: str, value_col: str):
    """날짜 컬럼 기준 기간별 집계. (granularity, labels, values, label_fmt)"""
    dt = pd.to_datetime(df[date_col].astype(str).str.strip(), errors="coerce", format="mixed")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    tmp = pd.DataFrame({"_d": dt, "_v": vals}).dropna()
    if tmp.empty:
        return None
    span_days = (tmp["_d"].max() - tmp["_d"].min()).days
    if span_days >= 60:
        gran, unit = "M", "월"
    elif span_days >= 14:
        gran, unit = "W", "주"
    else:
        gran, unit = "D", "일"
    g = tmp.groupby(pd.Grouper(key="_d", freq=gran))["_v"].sum().sort_index()
    labels, pretty = [], []
    for d in g.index:
        if gran == "M":
            labels.append(d.strftime("%Y-%m"))
            pretty.append(f"{d.year}년 {d.month}월")
        elif gran == "W":
            labels.append(d.strftime("%Y-%m-%d"))
            pretty.append(f"{d.month}월 {d.day}일 주")
        else:
            labels.append(d.strftime("%Y-%m-%d"))
            pretty.append(f"{d.month}월 {d.day}일")
    return {
        "granularity": gran, "unit": unit,
        "labels": labels, "pretty": pretty,
        "values": [float(v) for v in g.values],
    }


# ---------------------------------------------------------------- KPIs
def build_kpis(df: pd.DataFrame, coltypes: dict) -> list:
    primary = pick_primary_numeric(df, coltypes)
    if primary is None:
        return []
    vals = pd.to_numeric(df[primary], errors="coerce").dropna()
    date_col = pick_date_column(df, coltypes)
    kpis = [
        {"label": f"{primary} 총합", "value": f"{vals.sum():,.0f}",
         "sub": f"전체 {len(vals):,}건 집계", "tone": "neutral"},
        {"label": f"{primary} 평균", "value": f"{vals.mean():,.1f}",
         "sub": "건당 평균", "tone": "neutral"},
    ]
    if date_col:
        ps = period_series(df, date_col, primary)
        if ps and len(ps["values"]) >= 2:
            last, prev = ps["values"][-1], ps["values"][-2]
            unit = ps["unit"]
            mom = (last - prev) / prev * 100 if prev != 0 else 0.0
            arrow = "▲" if mom >= 0 else "▼"
            tone = "up" if mom >= 0 else "down"
            kpis.append({
                "label": f"전{unit} 대비 증감률", "value": f"{arrow} {abs(mom):.1f}%",
                "sub": f"{ps['pretty'][-2]} → {ps['pretty'][-1]}", "tone": tone,
                "_mom_pct": round(mom, 2),
            })
            bi = int(np.argmax(ps["values"]))
            kpis.append({
                "label": f"최고 {unit} ({ps['pretty'][bi]})",
                "value": f"{ps['values'][bi]:,.0f}",
                "sub": f"전체 기간 중 최대", "tone": "neutral",
            })
    else:
        kpis.append({"label": f"{primary} 최대값", "value": f"{vals.max():,.0f}",
                     "sub": "단일 최대 기록", "tone": "neutral"})
        kpis.append({"label": f"{primary} 최소값", "value": f"{vals.min():,.0f}",
                     "sub": "단일 최소 기록", "tone": "neutral"})
    return kpis


# ---------------------------------------------------------------- charts
def build_charts(df: pd.DataFrame, coltypes: dict) -> list:
    charts = []
    primary = pick_primary_numeric(df, coltypes)
    date_col = pick_date_column(df, coltypes)
    numerics = [c for c, t in coltypes.items() if t == "numeric"][:3]
    categories = [c for c, t in coltypes.items()
                  if t == "category" and df[c].nunique() <= MAX_CATEGORY_UNIQUE]

    # 1) 시계열 추이 — 대표 지표 단일 시계열 (스케일이 다른 지표 혼합 방지)
    if date_col and primary:
        ps = period_series(df, date_col, primary)
        if ps and len(ps["values"]) >= 2:
            charts.append({
                "id": "ch_timeseries", "type": "line",
                "title": f"기간별 {primary} 추이",
                "subtitle": "자동 인식된 날짜 컬럼 기준 집계",
                "data": {"labels": ps["pretty"], "datasets": [{
                    "label": str(primary), "data": ps["values"],
                    "borderColor": PALETTE[0],
                    "backgroundColor": PALETTE[0] + "2E",
                    "tension": 0.3, "fill": True, "pointRadius": 4,
                }]},
                "options": {"plugins": {"legend": {"display": False}}},
            })

    # 2) 범주별 막대
    if primary and categories:
        for ci, cat in enumerate(categories[:2]):
            grp = df.groupby(cat, observed=True)[primary].apply(
                lambda s: pd.to_numeric(s, errors="coerce").sum()
            ).sort_values(ascending=False).head(10)
            charts.append({
                "id": f"ch_bar_{ci}", "type": "bar",
                "title": f"{cat}별 {primary} 합계",
                "subtitle": "범주 자동 집계 (상위 10개)",
                "data": {
                    "labels": [str(x) for x in grp.index],
                    "datasets": [{
                        "label": str(primary),
                        "data": [float(v) for v in grp.values],
                        "backgroundColor": [PALETTE[i % len(PALETTE)] for i in range(len(grp))],
                    }],
                },
                "options": {"plugins": {"legend": {"display": False}}},
            })

    # 3) 파이 (첫 범주 컬럼)
    if primary and categories:
        cat = categories[0]
        grp = df.groupby(cat, observed=True)[primary].apply(
            lambda s: pd.to_numeric(s, errors="coerce").sum()
        ).sort_values(ascending=False)
        if len(grp) > 6:
            top = grp.head(5)
            other = grp.iloc[5:].sum()
            labels = [str(x) for x in top.index] + ["기타"]
            data = [float(v) for v in top.values] + [float(other)]
        else:
            labels = [str(x) for x in grp.index]
            data = [float(v) for v in grp.values]
        charts.append({
            "id": "ch_pie", "type": "pie",
            "title": f"{cat}별 {primary} 비중",
            "subtitle": "구성비 자동 계산",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                }],
            },
            "options": {"plugins": {"legend": {"position": "bottom"}}},
        })

    # 4) 분포 히스토그램
    for i, ncol in enumerate(numerics):
        vals = pd.to_numeric(df[ncol], errors="coerce").dropna()
        if len(vals) < 5:
            continue
        counts, edges = np.histogram(vals, bins=10)
        labels = [f"{edges[j]:,.0f}~{edges[j+1]:,.0f}" for j in range(len(counts))]
        charts.append({
            "id": f"ch_hist_{i}", "type": "bar",
            "title": f"{ncol} 분포",
            "subtitle": "히스토그램 (10구간)",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "건수",
                    "data": [int(c) for c in counts],
                    "backgroundColor": PALETTE[(i + 3) % len(PALETTE)] + "CC",
                }],
            },
            "options": {"plugins": {"legend": {"display": False}}},
        })
    return charts


# ---------------------------------------------------------------- insights
def _skew(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=float)
    m, s = v.mean(), v.std()
    if s == 0 or len(v) < 3:
        return 0.0
    return float(((v - m) ** 3).mean() / s ** 3)


def generate_insights(df: pd.DataFrame, coltypes: dict) -> list:
    """룰 기반 인사이트 생성. 전부 실제 데이터에서 계산된 수치만 사용."""
    insights = []
    primary = pick_primary_numeric(df, coltypes)
    if primary is None:
        return [{"level": "warning", "tag": "데이터 확인",
                 "title": "숫자형 컬럼이 없습니다",
                 "text": "업로드한 파일에서 숫자형 컬럼을 찾지 못했습니다. 매출·수량 등 숫자 컬럼이 포함된 CSV를 올려주세요.",
                 "data": {}}]
    date_col = pick_date_column(df, coltypes)
    categories = [c for c, t in coltypes.items()
                  if t == "category" and df[c].nunique() <= MAX_CATEGORY_UNIQUE]

    # 1) 전 기간 대비 증감 (헤드라인)
    if date_col:
        ps = period_series(df, date_col, primary)
        if ps and len(ps["values"]) >= 2:
            last, prev = ps["values"][-1], ps["values"][-2]
            unit = ps["unit"]
            mom = (last - prev) / prev * 100 if prev != 0 else 0.0
            direction = "증가" if mom >= 0 else "감소"
            insights.append({
                "level": "up" if mom >= 0 else "down",
                "tag": f"전{unit} 대비",
                "title": f"{ps['pretty'][-1]} {primary}, 전{unit} 대비 {abs(mom):.1f}% {direction}",
                "text": (f"{ps['pretty'][-1]} {primary}는 {last:,.0f}으로 "
                         f"전{unit}({prev:,.0f}) 대비 {abs(mom):.1f}% {direction}했습니다."),
                "data": {"mom_pct": round(mom, 2), "last": last, "prev": prev},
            })
            # 2) 추세
            v = np.array(ps["values"], dtype=float)
            x = np.arange(len(v))
            slope = float(np.polyfit(x, v, 1)[0])
            avg = v.mean()
            slope_pct = slope / avg * 100 if avg != 0 else 0.0
            if abs(slope_pct) < 0.5:
                t_text = (f"최근 {len(v)}{unit}간 {primary}는 뚜렷한 방향 없이 횡보하고 있습니다. "
                          f"(기간 평균 {avg:,.0f})")
                level = "info"
            else:
                t_dir = "상승" if slope_pct > 0 else "하락"
                t_text = (f"최근 {len(v)}{unit}간 {primary}는 {t_dir} 추세입니다. "
                          f"추세선 기준 {unit}평균 {abs(slope_pct):.1f}%씩 움직이고 있습니다.")
                level = "up" if slope_pct > 0 else "down"
            trend_word = t_dir if abs(slope_pct) >= 0.5 else "횡보"
            unit_noun = {"월": "개월", "주": "주", "일": "일"}.get(unit, unit)
            insights.append({
                "level": level, "tag": "추세 분석",
                "title": f"최근 {len(v)}{unit_noun}간 추세: {trend_word}",
                "text": t_text,
                "data": {"slope_pct": round(slope_pct, 2)},
            })
            # 3) 이상치 탐지 (Z-score)
            std = v.std(ddof=1) if len(v) > 2 else 0
            if std > 0:
                z = (v - v.mean()) / std
                hits = [(ps["pretty"][i], v[i], float(z[i]))
                        for i in range(len(v)) if abs(z[i]) >= ANOMALY_Z_THRESHOLD]
                if hits:
                    parts = []
                    for label, val, zi in hits:
                        diff = (val - v.mean()) / v.mean() * 100 if v.mean() != 0 else 0
                        up_down = "높은" if zi > 0 else "낮은"
                        parts.append(f"{label}({val:,.0f}, 평균 대비 {abs(diff):.0f}% {up_down})")
                    insights.append({
                        "level": "warning", "tag": "이상치 탐지",
                        "title": f"{len(hits)}개 기간에서 이상치 감지 (Z-score ≥ {ANOMALY_Z_THRESHOLD})",
                        "text": ("다음 기간의 " + primary + "가 통계적 이상치로 감지되었습니다: "
                                 + ", ".join(parts) + ". 원인 파악이 필요합니다."),
                        "data": {"anomalies": [{"label": h[0], "value": h[1], "z": round(h[2], 2)} for h in hits]},
                    })
                else:
                    insights.append({
                        "level": "info", "tag": "이상치 탐지",
                        "title": "이상치 없음",
                        "text": (f"전체 {len(v)}{unit} 중 Z-score {ANOMALY_Z_THRESHOLD} 이상인 이상치는 "
                                 f"감지되지 않았습니다. {primary} 흐름이 안정적입니다."),
                        "data": {"anomalies": []},
                    })

    # 4) 상위/하위 항목 요약
    if categories:
        cat = categories[0]
        grp = df.groupby(cat, observed=True)[primary].apply(
            lambda s: pd.to_numeric(s, errors="coerce").sum()
        ).sort_values(ascending=False)
        total = grp.sum()
        if total != 0 and len(grp) >= 2:
            top1, top2 = grp.index[0], grp.index[1]
            bot = grp.index[-1]
            insights.append({
                "level": "info", "tag": "항목별 요약",
                "title": f"{cat} 기준 1위: {top1} ({grp.iloc[0]/total*100:.1f}%)",
                "text": (f"{cat}별 {primary} 합계 기준 상위 항목은 {top1}({grp.iloc[0]/total*100:.1f}%), "
                         f"{top2}({grp.iloc[1]/total*100:.1f}%)이며, "
                         f"하위 항목은 {bot}({grp.iloc[-1]/total*100:.1f}%)입니다."),
                "data": {"top1": str(top1), "top1_share": round(grp.iloc[0]/total*100, 1),
                         "bottom": str(bot), "bottom_share": round(grp.iloc[-1]/total*100, 1)},
            })
        # 5) 채널/두 번째 범주 비교
        if len(categories) >= 2:
            cat2 = categories[1]
            grp2 = df.groupby(cat2, observed=True)[primary].apply(
                lambda s: pd.to_numeric(s, errors="coerce").sum()
            ).sort_values(ascending=False)
            if len(grp2) >= 2 and grp2.iloc[-1] != 0:
                ratio = grp2.iloc[0] / grp2.iloc[-1]
                insights.append({
                    "level": "info", "tag": "그룹 비교",
                    "title": f"{grp2.index[0]}이(가) {grp2.index[-1]}의 {ratio:.1f}배",
                    "text": (f"{cat2}별 {primary} 합계는 {grp2.index[0]}({grp2.iloc[0]:,.0f})이 "
                             f"가장 크며, {grp2.index[-1]}({grp2.iloc[-1]:,.0f})의 "
                             f"약 {ratio:.1f}배에 달합니다."),
                    "data": {"ratio": round(ratio, 2)},
                })

    # 6) 분포 왜도
    vals = pd.to_numeric(df[primary], errors="coerce").dropna().values
    if len(vals) >= 10:
        sk = _skew(vals)
        if sk > 0.5:
            d_text = (f"{primary} 분포는 오른쪽으로 치우쳐 있습니다(왜도 {sk:.2f}). "
                      f"일부 큰 값이 평균({vals.mean():,.0f})을 끌어올리고 있습니다.")
        elif sk < -0.5:
            d_text = (f"{primary} 분포는 왼쪽으로 치우쳐 있습니다(왜도 {sk:.2f}). "
                      f"일부 작은 값이 평균({vals.mean():,.0f})을 끌어내리고 있습니다.")
        else:
            d_text = (f"{primary} 분포는 비교적 대칭적입니다(왜도 {sk:.2f}). "
                      f"평균 {vals.mean():,.0f}이 대표값으로 적절합니다.")
        insights.append({
            "level": "info", "tag": "분포 분석",
            "title": f"{primary} 분포 왜도 {sk:.2f}",
            "text": d_text, "data": {"skew": round(sk, 2)},
        })

    # 7) 상관관계 (두 번째 숫자 컬럼과)
    numerics = [c for c, t in coltypes.items() if t == "numeric" and c != primary]
    if numerics:
        other = None
        for c in numerics:
            if any(k in str(c) for k in ["광고", "비용", "마케팅", "ad", "cost"]):
                other = c
                break
        other = other or numerics[0]
        a = pd.to_numeric(df[primary], errors="coerce")
        b = pd.to_numeric(df[other], errors="coerce")
        both = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(both) >= 5:
            r = float(both["a"].corr(both["b"]))
            if abs(r) >= 0.7:
                c_text = (f"{other}와(과) {primary}의 상관계수는 {r:.2f}로 강한 "
                          f"{'양' if r > 0 else '음'}의 상관관계가 있습니다.")
            elif abs(r) >= 0.4:
                c_text = (f"{other}와(과) {primary}의 상관계수는 {r:.2f}로 중간 정도의 "
                          f"{'양' if r > 0 else '음'}의 상관관계가 있습니다.")
            else:
                c_text = (f"{other}와(과) {primary}의 상관계수는 {r:.2f}로 뚜렷한 "
                          f"선형 관계가 보이지 않습니다.")
            insights.append({
                "level": "info", "tag": "상관 분석",
                "title": f"{other} ↔ {primary} 상관계수 {r:.2f}",
                "text": c_text, "data": {"corr": round(r, 2), "other": str(other)},
            })

    return insights


def analyze(df: pd.DataFrame) -> dict:
    """전체 파이프라인: 컬럼 인식 → KPI → 차트 → 인사이트."""
    coltypes = detect_columns(df)
    return {
        "columns": [{"name": str(c), "type": t} for c, t in coltypes.items()],
        "kpis": build_kpis(df, coltypes),
        "charts": build_charts(df, coltypes),
        "insights": generate_insights(df, coltypes),
        "n_rows": len(df),
    }
