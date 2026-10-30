#!/usr/bin/env python3
"""Rebuild every chart and data file in this repository from the UN source data.

Source: UN DESA Population Division, World Population Prospects 2024,
Demographic Indicators file (medium variant), column LEx (life expectancy at
birth, both sexes, years).

    python scripts/build_charts.py

The raw UN file (~16 MB) is downloaded once to data/raw/ and reused. Outputs:

    charts/flat-life-expectancy-mao-era.html
    charts/china-life-expectancy-vs-trend-1950-76.html
    charts/life-expectancy-stalls-russia-zimbabwe.html
    data/life-expectancy-gains-1950-1976-un-wpp2024.csv

Each chart is a self-contained HTML page: scripts/chart_template.html holds the
renderer, and this script injects the title and the CHART config (the data).
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "WPP2024_Demographic_Indicators_Medium.csv.gz"
TEMPLATE = ROOT / "scripts" / "chart_template.html"
WPP_URL = ("https://population.un.org/wpp/assets/Excel%20Files/"
           "1_Indicator%20(Standard)/CSV_FILES/"
           "WPP2024_Demographic_Indicators_Medium.csv.gz")

SOURCE = ("UN DESA Population Division, World Population Prospects 2024, "
          "Demographic Indicators file (medium variant), column LEx")
SOURCE_URL = ("https://population.un.org/wpp/downloads?folder="
              "Standard%20Projections&group=CSV%20format")

START, END = 1950, 1976          # Mao-era window used in the first two charts
MIN_POP_1950 = 1000              # thousands, i.e. 1m+ people in 1950
# Excluded from the scatter and the fit because their 1950 or 1976 value is
# dominated by a war or genocide rather than by development.
PRK = "Dem. People's Republic of Korea"
EXCLUDE = ["Cambodia", "Lebanon", "Republic of Korea", PRK]

# UN names that the charts shorten for display.
SHORT = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Democratic Republic of the Congo": "DR Congo",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "China, Taiwan Province of China": "Taiwan",
    "United Republic of Tanzania": "Tanzania",
    "Türkiye": "Turkey",
    "United States of America": "United States",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
}

CSV_COLUMNS = ["Location", "LocID", "ISO3_code", "le_1950", "le_1976", "gain_1950_76",
               "pop_1950_thousands", "in_scatter", "fitted_gain"]

LABELLED = ["India", "Pakistan", "Indonesia", "Brazil", "Mexico", "Niger",
            "Rwanda", "Burundi", "Somalia", "Netherlands", "Japan", "Sri Lanka",
            "Nigeria", "Egypt", "Taiwan", "Russia", "United States",
            "Ethiopia", "Philippines"]


def load():
    """Medium-variant life expectancy and mid-year population, one row per
    location and year. Locations are countries/areas plus the World total."""
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", WPP_URL)
        urllib.request.urlretrieve(WPP_URL, RAW)
    d = pd.read_csv(RAW, low_memory=False, encoding="utf-8-sig")
    d = d[(d.Variant == "Medium")
          & d.LocTypeName.isin(["Country/Area", "World"])]
    return d[["Location", "LocID", "ISO3_code", "LocTypeName", "Time", "LEx",
              "TPopulation1July"]]


def series(d, name, y0, y1, color):
    s = d[(d.Location == name) & d.Time.between(y0, y1)].sort_values("Time")
    return {"name": name, "color": color,
            "data": [[int(t), round(float(v), 2)] for t, v in zip(s.Time, s.LEx)]}


def value(d, name, year):
    return round(float(d[(d.Location == name) & (d.Time == year)].LEx.iloc[0]), 2)


def first_year_above(s, level, after):
    return next(y for y, v in s["data"] if y > after and v > level)


def sgn(v):
    return f"+{v:g}" if v >= 0 else f"−{abs(v):g}"


def r2(x):
    """Round to 2 dp with Python's round(), as the series do (pandas rounds halves differently)."""
    return pd.Series(x).map(lambda v: round(float(v), 2)).values


def gains_table(d):
    """One row per country: LEx in 1950 and 1976, the change, and the OLS fit of
    change on starting LEx across the countries that go in the scatter."""
    c = d[d.LocTypeName == "Country/Area"]
    a = c[c.Time == START].set_index("Location")
    b = c[c.Time == END].set_index("Location")
    le76 = b.LEx.reindex(a.index)
    t = pd.DataFrame({
        "Location": a.index, "LocID": a.LocID.values, "ISO3_code": a.ISO3_code.values,
        "le_1950": r2(a.LEx), "le_1976": r2(le76),
        "pop_1950_thousands": r2(a.TPopulation1July)})
    # Changes use the UN's unrounded values; the fit uses the rounded values as plotted.
    t["gain_raw"] = (le76 - a.LEx).values
    t["gain_1950_76"] = r2(t.gain_raw)
    t["in_scatter"] = (t.pop_1950_thousands >= MIN_POP_1950) & ~t.Location.isin(EXCLUDE)
    sc = t[t.in_scatter]
    slope, icpt = np.polyfit(sc.le_1950, sc.gain_1950_76, 1)
    t["fitted_raw"] = np.where(t.in_scatter, icpt + slope * t.le_1950, np.nan)
    t["fitted_gain"] = r2(t.fitted_raw)
    return t.sort_values("gain_1950_76", kind="stable").reset_index(drop=True)


def chart_flat(d, g):
    countries = [("China", "#C71E1D"), ("India", "#1B3A5C"),
                 ("Rwanda", "#2E8B7A"), ("Niger", "#E8A33D")]
    gain = g.set_index("Location").gain_1950_76.round(1)
    china_low = value(d, "China", 1960)
    return {
        "type": "line",
        "headline": "Over Mao's era, Niger and Rwanda gained barely a year or two of life expectancy",
        "subhead": f"Life expectancy at birth, years, {START}–{END}",
        "source": SOURCE, "sourceUrl": SOURCE_URL,
        "note": (f"Change {START}–76: China {sgn(gain['China'])}, India {sgn(gain['India'])}, "
                 f"Rwanda {sgn(gain['Rwanda'])}, Niger {sgn(gain['Niger'])} years. "
                 f"Burundi ({sgn(gain['Burundi'])}) and Somalia ({sgn(gain['Somalia'])}) were similar. "
                 "UN figures for these countries in the 1950s and 1960s rest on few surveys "
                 "and are model-based."),
        "xType": "year", "format": {"decimals": 0},
        "series": [series(d, n, START, END, col) for n, col in countries],
        "annotations": [{"x": 1960, "y": china_low,
                         "text": f"Great Leap famine: {china_low:.1f} in 1960",
                         "dx": 24, "dy": 10, "series": "China"}],
    }


def chart_scatter(g):
    sc = g[g.in_scatter].copy()
    n = len(sc)
    china = sc[sc.Location == "China"].iloc[0]
    above = china.gain_raw - china.fitted_raw
    gain = g.set_index("Location").gain_1950_76
    sc["label"] = sc.Location.replace(SHORT)
    sc = sc.sort_values("Location", kind="stable")

    def pts(rows):
        return [[float(r.le_1950), float(r.gain_1950_76), r.label]
                for r in rows.itertuples()]

    return {
        "type": "scatter",
        "headline": "China's life expectancy gains under Mao beat the trend for its starting point",
        "subhead": (f"Life expectancy at birth in {START} vs change from {START} to {END}, years. "
                    f"{n} countries with 1m+ people in {START}"),
        "source": SOURCE, "sourceUrl": SOURCE_URL,
        "note": (f"Dashed line is the OLS fit across all {n} countries. Excludes Cambodia "
                 f"({sgn(round(gain['Cambodia']))} years) and Lebanon ({sgn(round(gain['Lebanon']))}), "
                 "whose 1976 values reflect the Khmer Rouge and the civil war, and North and South "
                 f"Korea ({sgn(round(gain[PRK]))}, "
                 f"{sgn(round(gain['Republic of Korea']))}), whose 1950 values reflect the Korean War. "
                 "China's 1950s values are UN reconstructions from sparse data; its 1960 famine "
                 "low (33.4) is inside this window."),
        "xType": "number", "xLabel": f"Life expectancy in {START} (years)",
        "yLabel": f"Change, {START} to {END} (years)",
        "format": {"decimals": 0}, "xFormat": {"decimals": 1}, "fit": True,
        "series": [
            {"name": "Other countries", "color": "#6B7280",
             "data": pts(sc[sc.Location != "China"])},
            {"name": "China", "color": "#C71E1D", "data": pts(sc[sc.Location == "China"])},
        ],
        "pointLabels": LABELLED,
        "annotations": [{"x": float(china.le_1950), "y": float(china.gain_1950_76),
                         "text": f"China: {sgn(float(china.gain_1950_76))} years, "
                                 f"{above:.1f} above the line",
                         "dx": 40, "dy": -46, "series": "China"}],
    }


def chart_stalls(d):
    y0, y1 = 1953, 2023
    ru = series(d, "Russian Federation", y0, y1, "#C71E1D")
    ru["name"] = "Russia"
    zw = series(d, "Zimbabwe", y0, y1, "#E8A33D")
    cn = series(d, "China", y0, y1, "#1B3A5C")
    wd = series(d, "World", y0, y1, "#6B7280")
    rd, zd = dict(ru["data"]), dict(zw["data"])
    ru_peak_y = max(range(1986, 1992), key=lambda y: rd[y])         # late-1980s peak
    ru_back = first_year_above(ru, rd[ru_peak_y], ru_peak_y)
    zw_1986 = zd[1986]
    zw_low_y = min(range(1990, 2011), key=lambda y: zd[y])
    zw_back = first_year_above(zw, zw_1986, zw_low_y)
    return {
        "type": "line",
        "headline": "Life expectancy can stall for decades, as it did in Russia and Zimbabwe",
        "subhead": f"Life expectancy at birth, years, {y0}–{y1}",
        "source": SOURCE, "sourceUrl": SOURCE_URL,
        "note": (f"Russia: {rd[1964]:.1f} in 1964, {rd[1985]:.1f} in 1985; after a late-1980s peak of "
                 f"{rd[ru_peak_y]:.1f} it did not pass that level again until {ru_back}. "
                 f"Zimbabwe: {zw_1986:.1f} in 1986, a low of {zd[zw_low_y]:.1f} in {zw_low_y} during "
                 f"the HIV epidemic, and back above its 1986 level in {zw_back}."),
        "xType": "year", "format": {"decimals": 0},
        "series": [ru, zw, cn, wd],
    }


def render(title, chart):
    html = TEMPLATE.read_text(encoding="utf-8")
    cfg = json.dumps(chart, indent=2, ensure_ascii=False)
    return html.replace("{{TITLE}}", title).replace("{{CHART_JSON}}", cfg)


def main():
    d = load()
    g = gains_table(d)
    charts = {
        "flat-life-expectancy-mao-era.html": chart_flat(d, g),
        "china-life-expectancy-vs-trend-1950-76.html": chart_scatter(g),
        "life-expectancy-stalls-russia-zimbabwe.html": chart_stalls(d),
    }
    out = ROOT / "charts"
    out.mkdir(exist_ok=True)
    for fname, ch in charts.items():
        (out / fname).write_text(render(ch["headline"], ch), encoding="utf-8")
        print("wrote", out / fname)
    csv = ROOT / "data" / "life-expectancy-gains-1950-1976-un-wpp2024.csv"
    g[CSV_COLUMNS].to_csv(csv, index=False)
    print("wrote", csv)


if __name__ == "__main__":
    main()
