# -*- coding: utf-8 -*-
"""External surveillance comparison for the SEIR V4/V5 project.

This script does not refit or modify either SEIR model. It aligns an external
weekly China CDC sentinel SARS-CoV-2 positivity series with V4 daily observed
and predicted cases, and uses V5 only to display change-point uncertainty.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ORIGINAL_CHANGE_POINT = pd.Timestamp("2025-05-24")
LAGS = [-2, -1, 0, 1, 2]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positivity", type=Path, default=here / "Positive rate of virus.csv",
        help="Sentinel positivity CSV or Excel file (default: Positive rate of virus.csv).",
    )
    parser.add_argument(
        "--predictions", type=Path, default=here / "seir_output_v4" / "predictions.csv",
        help="V4 predictions.csv.",
    )
    parser.add_argument(
        "--bootstrap", type=Path,
        default=here / "seir_output_v5" / "bootstrap_results.csv",
        help="V5 bootstrap_results.csv.",
    )
    parser.add_argument(
        "--output", type=Path, default=here / "external_validation",
        help="Independent output directory.",
    )
    return parser.parse_args()


def normalized_name(value) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).strip().lower())


def choose_column(columns, groups, exclude=()):
    excluded = set(exclude)
    normalized = {c: normalized_name(c) for c in columns if c not in excluded}
    for tokens in groups:
        normalized_tokens = [normalized_name(t) for t in tokens]
        matches = [c for c, name in normalized.items()
                   if all(t in name for t in normalized_tokens)]
        if matches:
            return matches[0]
    return None


def parse_date_value(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(parsed).normalize() if not pd.isna(parsed) else pd.NaT


def parse_week_interval(value, default_year=2025):
    """Parse common English/Chinese week ranges into inclusive start/end dates."""
    if pd.isna(value):
        return pd.NaT, pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        start = pd.Timestamp(value).normalize()
        return start, start + pd.Timedelta(days=6)

    text = str(value).strip()
    cleaned = (text.replace("年", "-").replace("月", "-").replace("日", "")
               .replace("至", "-").replace("—", "-").replace("–", "-")
               .replace("~", "-").replace("to", "-"))

    # First try to identify two complete date-like substrings.
    full_dates = re.findall(r"(?:20\d{2}[-/.])?\d{1,2}[-/.]\d{1,2}", cleaned)
    parsed = []
    for item in full_dates[:2]:
        bits = [int(x) for x in re.split(r"[-/.]", item)]
        if len(bits) == 3:
            year, month, day = bits
        else:
            year, month, day = default_year, bits[0], bits[1]
        try:
            parsed.append(pd.Timestamp(year=year, month=month, day=day))
        except ValueError:
            pass
    if len(parsed) == 2:
        if parsed[1] < parsed[0] and parsed[1].month < parsed[0].month:
            parsed[1] = parsed[1].replace(year=parsed[0].year + 1)
        return parsed[0], parsed[1]

    # Handles abbreviated ranges such as "2025-05-19 - 25" or "May 19-25, 2025".
    nums = [int(x) for x in re.findall(r"\d+", cleaned)]
    abbreviated_numeric = re.search(
        r"(?:(20\d{2})[-/.])?(\d{1,2})[-/.](\d{1,2})\s*-\s*(\d{1,2})$",
        cleaned,
    )
    if abbreviated_numeric:
        year = int(abbreviated_numeric.group(1) or default_year)
        month = int(abbreviated_numeric.group(2))
        start_day = int(abbreviated_numeric.group(3))
        end_day = int(abbreviated_numeric.group(4))
        try:
            return (pd.Timestamp(year, month, start_day),
                    pd.Timestamp(year, month, end_day))
        except ValueError:
            pass

    month_names = {m.lower(): i for i, m in enumerate(
        ["", "January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    )}
    month_match = re.search("|".join(month_names.keys() - {""}), text.lower())
    if month_match:
        month = month_names[month_match.group(0)]
        days = [x for x in nums if x != default_year and 1 <= x <= 31]
        if len(days) >= 2:
            return (pd.Timestamp(default_year, month, days[0]),
                    pd.Timestamp(default_year, month, days[1]))

    # A generic date cell is interpreted as a week start only after all range
    # formats above fail. This rule is reported in the run summary.
    single_date = parse_date_value(value)
    if not pd.isna(single_date):
        return single_date, single_date + pd.Timedelta(days=6)
    return pd.NaT, pd.NaT


def read_positivity_table(path: Path) -> pd.DataFrame:
    """Read CSV without optional Excel packages; retain Excel support if installed."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        errors = []
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
        raise ValueError(
            "Could not decode the positivity CSV as UTF-8 or GB18030.\n"
            + "\n".join(errors)
        )
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx requires openpyxl. Either install it with "
                "`python -m pip install openpyxl`, or save the workbook as "
                "CSV UTF-8 and pass --positivity \"Positive rate of virus.csv\"."
            ) from exc
    if suffix == ".xls":
        raise ValueError(
            "Legacy .xls is not supported without another optional package. "
            "Please save it as CSV UTF-8 or .xlsx."
        )
    raise ValueError(f"Unsupported positivity file type: {suffix}. Use .csv or .xlsx.")


def load_positivity(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Positivity workbook not found: {path.resolve()}")
    raw = read_positivity_table(path)

    # Deliberately print the exact raw names before normalization: spaces and
    # capitalization in the workbook must be visible to the analyst.
    print(f"Raw columns read from {path.name!r}:")
    for i, column in enumerate(raw.columns, 1):
        print(f"  {i}. {column!r}")

    columns = list(raw.columns)
    pos_col = choose_column(columns, [
        ("positivity",), ("positiverate",), ("positive", "rate"),
        ("sarscov2", "rate"), ("covid", "rate"), ("阳性率",),
    ])
    if pos_col is None:
        numeric_candidates = []
        for col in columns:
            converted = pd.to_numeric(raw[col], errors="coerce")
            if converted.notna().sum() >= max(3, len(raw) // 2):
                numeric_candidates.append(col)
        if len(numeric_candidates) == 1:
            pos_col = numeric_candidates[0]
        else:
            raise ValueError(
                "Could not uniquely identify the positivity-rate column. "
                f"Numeric candidates: {numeric_candidates}"
            )

    start_col = choose_column(columns, [
        ("week", "start"), ("start", "date"), ("start",), ("开始",), ("起始",),
    ], exclude=[pos_col])
    end_col = choose_column(columns, [
        ("week", "end"), ("end", "date"), ("end",), ("结束",), ("截止",),
    ], exclude=[pos_col, start_col])

    week_col = None
    if start_col is None or end_col is None:
        week_col = choose_column(columns, [
            ("week", "range"), ("week", "interval"), ("week",),
            ("date", "range"), ("date",), ("周",), ("日期",),
        ], exclude=[pos_col])

    out = pd.DataFrame(index=raw.index)
    if start_col is not None and end_col is not None:
        out["Week_start"] = raw[start_col].map(parse_date_value)
        out["Week_end"] = raw[end_col].map(parse_date_value)
        date_source = f"explicit columns {start_col!r} and {end_col!r}"
    elif week_col is not None:
        intervals = raw[week_col].map(parse_week_interval)
        out["Week_start"] = [x[0] for x in intervals]
        out["Week_end"] = [x[1] for x in intervals]
        date_source = f"interval/date column {week_col!r}"
    elif start_col is not None:
        out["Week_start"] = raw[start_col].map(parse_date_value)
        out["Week_end"] = out["Week_start"] + pd.Timedelta(days=6)
        date_source = f"week-start column {start_col!r} (end inferred as start + 6 days)"
    elif end_col is not None:
        out["Week_end"] = raw[end_col].map(parse_date_value)
        out["Week_start"] = out["Week_end"] - pd.Timedelta(days=6)
        date_source = f"week-end column {end_col!r} (start inferred as end - 6 days)"
    else:
        raise ValueError("Could not identify a week/date column in the positivity workbook.")

    raw_pos_text = raw[pos_col].astype(str)
    has_percent_sign = raw_pos_text.str.contains(r"[%％]", regex=True, na=False)
    cleaned_pos = raw_pos_text.str.replace(r"[%％,，\s]", "", regex=True)
    out["Positivity_raw"] = pd.to_numeric(cleaned_pos, errors="coerce")
    bad = out[["Week_start", "Week_end", "Positivity_raw"]].isna().any(axis=1)
    exclusions = [f"Input row {i + 2}: missing/unparseable week or positivity"
                  for i in out.index[bad]]
    out = out.loc[~bad].copy()
    if out.empty:
        raise ValueError("No usable positivity observations remained after validation.")
    if (out["Week_end"] < out["Week_start"]).any():
        raise ValueError("At least one positivity week ends before it starts.")
    if out.duplicated(["Week_start", "Week_end"]).any():
        dup = out.loc[out.duplicated(["Week_start", "Week_end"], keep=False)]
        raise ValueError(f"Duplicate positivity weeks found:\n{dup}")
    if (out["Positivity_raw"] < 0).any():
        raise ValueError("Positivity rates cannot be negative.")

    # If values exceed 1, treat them as percentages and divide by 100 for analysis.
    retained_percent_sign = has_percent_sign.loc[out.index]
    if retained_percent_sign.any():
        if not retained_percent_sign.all():
            raise ValueError("Mixed percentage-sign and non-percentage positivity values found.")
        out["Positivity_rate"] = out["Positivity_raw"] / 100.0
        positivity_scale = "percentage strings; percent sign removed and values divided by 100"
    elif out["Positivity_raw"].max() <= 1.0:
        out["Positivity_rate"] = out["Positivity_raw"]
        positivity_scale = "proportion (0-1); multiplied by 100 only for display"
    elif out["Positivity_raw"].max() <= 100.0:
        out["Positivity_rate"] = out["Positivity_raw"] / 100.0
        positivity_scale = "percentage (0-100); divided by 100 for analysis"
    else:
        raise ValueError("Positivity values exceed 100 and cannot be interpreted as rates.")
    out["Positivity_percent"] = 100.0 * out["Positivity_rate"]
    out = out.sort_values("Week_start").reset_index(drop=True)
    return out, exclusions, pos_col, date_source, positivity_scale


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"V4 predictions file not found: {path.resolve()}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    columns = list(raw.columns)
    date_col = choose_column(columns, [("date",)])
    observed_col = choose_column(columns, [
        ("observed", "daily", "cases"), ("observed", "cases"), ("dailycases",),
    ])
    predicted_col = choose_column(columns, [
        ("freetc", "piecewise", "predicted"), ("piecewise", "predicted"),
        ("seir", "predicted"), ("predicted",),
    ], exclude=[observed_col])
    if None in (date_col, observed_col, predicted_col):
        raise ValueError(
            "Could not identify Date, observed cases and free-change-point prediction columns. "
            f"Columns: {columns}"
        )
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw[date_col], errors="coerce"),
        "Observed": pd.to_numeric(raw[observed_col], errors="coerce"),
        "Predicted": pd.to_numeric(raw[predicted_col], errors="coerce"),
    })
    if out.isna().any().any():
        raise ValueError("V4 predictions contain missing or unparseable required values.")
    if out["Date"].duplicated().any():
        raise ValueError("V4 predictions contain duplicate dates.")
    return out.sort_values("Date").reset_index(drop=True), {
        "date": date_col, "observed": observed_col, "predicted": predicted_col,
    }


def load_bootstrap_change_points(path: Path, study_start: pd.Timestamp):
    if not path.exists():
        raise FileNotFoundError(f"V5 bootstrap results not found: {path.resolve()}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    success_col = choose_column(raw.columns, [("success",)])
    if success_col is not None:
        success = raw[success_col].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        raw = raw.loc[success].copy()
    date_col = choose_column(raw.columns, [
        ("change", "date"), ("tcdate",), ("changepoint",), ("tc",),
    ])
    index_col = choose_column(raw.columns, [
        ("change", "index"), ("tcday",), ("change", "day"),
    ], exclude=[date_col])
    change_dates = None
    if date_col is not None:
        change_dates = pd.to_datetime(raw[date_col], errors="coerce")
    if (change_dates is None or change_dates.notna().sum() == 0) and index_col is not None:
        indices = pd.to_numeric(raw[index_col], errors="coerce")
        change_dates = study_start + pd.to_timedelta(indices, unit="D")
    if change_dates is None:
        raise ValueError(f"Could not identify bootstrap change points. Columns: {list(raw.columns)}")
    change_dates = pd.Series(change_dates).dropna().sort_values().reset_index(drop=True)
    if change_dates.empty:
        raise ValueError("No successful, parseable V5 change points were found.")
    day_values = (change_dates - study_start).dt.days.to_numpy()
    q_low, q_median, q_high = np.percentile(day_values, [2.5, 50, 97.5])
    return {
        "n": len(change_dates),
        "low": study_start + pd.Timedelta(days=int(np.floor(q_low))),
        "median": study_start + pd.Timedelta(days=int(np.rint(q_median))),
        "high": study_start + pd.Timedelta(days=int(np.ceil(q_high))),
    }


def align_weekly(positivity: pd.DataFrame, daily: pd.DataFrame):
    indexed = daily.set_index("Date")
    rows, exclusions = [], []
    for _, week in positivity.iterrows():
        expected_dates = pd.date_range(week["Week_start"], week["Week_end"], freq="D")
        available = indexed.reindex(expected_dates)
        if available[["Observed", "Predicted"]].isna().any().any():
            exclusions.append(
                f"{week['Week_start'].date()} to {week['Week_end'].date()}: "
                "daily observed/predicted data do not cover the full stated interval"
            )
            continue
        rows.append({
            "Week_start": week["Week_start"], "Week_end": week["Week_end"],
            "Week_midpoint": week["Week_start"] + (week["Week_end"] - week["Week_start"]) / 2,
            "n_days": len(expected_dates), "Positivity_rate": week["Positivity_rate"],
            "Positivity_percent": week["Positivity_percent"],
            "Observed_cases_sum": available["Observed"].sum(),
            "Observed_cases_mean": available["Observed"].mean(),
            "Predicted_cases_sum": available["Predicted"].sum(),
            "Predicted_cases_mean": available["Predicted"].mean(),
        })
    aligned = pd.DataFrame(rows)
    if len(aligned) < 3:
        raise ValueError("Fewer than three complete overlapping weeks are available.")

    # IMPORTANT: all three z-scores use these overlapping weeks only. They are
    # not standardized over the full length of each original series.
    for source, target in [
        ("Observed_cases_mean", "Observed_cases_z"),
        ("Predicted_cases_mean", "Predicted_cases_z"),
        ("Positivity_rate", "Positivity_z"),
    ]:
        sd = aligned[source].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot compute a z-score for constant series {source}.")
        aligned[target] = (aligned[source] - aligned[source].mean()) / sd
    return aligned, exclusions


def spearman_pair(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan, len(valid)
    result = spearmanr(valid["x"], valid["y"])
    return float(result.statistic), float(result.pvalue), len(valid)


def lagged_pair(positivity, cases, lag):
    """Positive lag means positivity at week t is compared with cases at t+lag."""
    p = np.asarray(positivity)
    c = np.asarray(cases)
    if lag > 0:
        return p[:-lag], c[lag:]
    if lag < 0:
        k = -lag
        return p[k:], c[:-k]
    return p, c


def correlation_outputs(aligned):
    rows = []
    for label, col in [
        ("Sentinel positivity vs weekly observed cases", "Observed_cases_mean"),
        ("Sentinel positivity vs weekly SEIR-predicted cases", "Predicted_cases_mean"),
    ]:
        rho, p, n = spearman_pair(aligned["Positivity_rate"], aligned[col])
        rows.append({"comparison": label, "rho": rho, "p_value": p, "n_weeks": n})
    return pd.DataFrame(rows)


def first_difference_outputs(aligned):
    """Test short-term co-movement after removing the shared level trend."""
    positivity_diff = np.diff(aligned["Positivity_rate"].to_numpy(dtype=float))
    observed_diff = np.diff(aligned["Observed_cases_mean"].to_numpy(dtype=float))
    predicted_diff = np.diff(aligned["Predicted_cases_mean"].to_numpy(dtype=float))

    rho_observed, p_observed, n_observed = spearman_pair(
        positivity_diff, observed_diff
    )
    rho_predicted, p_predicted, n_predicted = spearman_pair(
        positivity_diff, predicted_diff
    )
    return {
        "rho_observed": rho_observed,
        "p_observed": p_observed,
        "n_observed": n_observed,
        "rho_predicted": rho_predicted,
        "p_predicted": p_predicted,
        "n_predicted": n_predicted,
    }


def lag_outputs(aligned):
    rows = []
    for lag in LAGS:
        p_obs, obs = lagged_pair(aligned["Positivity_rate"], aligned["Observed_cases_mean"], lag)
        p_pred, pred = lagged_pair(aligned["Positivity_rate"], aligned["Predicted_cases_mean"], lag)
        rho_o, p_o, n_o = spearman_pair(p_obs, obs)
        rho_p, p_p, n_p = spearman_pair(p_pred, pred)
        rows.append({
            "lag_weeks": lag, "rho_observed": rho_o, "p_observed": p_o,
            "n_observed": n_o, "rho_predicted": rho_p, "p_predicted": p_p,
            "n_predicted": n_p,
        })
    return pd.DataFrame(rows)


def fmt_week(start, end):
    if start.year == end.year and start.month == end.month:
        return f"{start:%b %d}-{end:%d}, {start:%Y}"
    if start.year == end.year:
        return f"{start:%b %d}-{end:%b %d}, {start:%Y}"
    return f"{start:%b %d, %Y}-{end:%b %d, %Y}"


def make_turning_points(daily, positivity, bootstrap):
    observed_peak = pd.Timestamp(daily.loc[daily["Observed"].idxmax(), "Date"])
    predicted_peak = pd.Timestamp(daily.loc[daily["Predicted"].idxmax(), "Date"])
    sentinel_peak = positivity.loc[positivity["Positivity_rate"].idxmax()]
    peak_start, peak_end = sentinel_peak["Week_start"], sentinel_peak["Week_end"]
    in_peak = bool(peak_start <= ORIGINAL_CHANGE_POINT <= peak_end)
    comparison_text = (
        f"May 24 (SEIR change point) falls within the sentinel positivity peak week "
        f"({peak_start:%b %d}-{peak_end:%d}): {str(in_peak).upper()}"
    )
    rows = [
        {"Indicator": "SEIR original change point", "Timing": "May 24, 2025",
         "Interpretation": "Original-data point estimate; not replaced by bootstrap results."},
        {"Indicator": "SEIR model-predicted peak", "Timing": f"{predicted_peak:%b %d, %Y}",
         "Interpretation": "Automatically identified from V4 daily predictions."},
        {"Indicator": "Observed reported-case peak", "Timing": f"{observed_peak:%b %d, %Y}",
         "Interpretation": "Automatically identified from V4 daily observations."},
        {"Indicator": "Sentinel positivity peak", "Timing": fmt_week(peak_start, peak_end),
         "Interpretation": comparison_text},
        {"Indicator": "May 24 within sentinel positivity peak week", "Timing": str(in_peak).upper(),
         "Interpretation": comparison_text},
        {"Indicator": "Bootstrap median change point", "Timing": f"{bootstrap['median']:%b %d, %Y}",
         "Interpretation": f"Based on {bootstrap['n']} successful V5 replicates."},
        {"Indicator": "Bootstrap 95% change-point interval",
         "Timing": f"{bootstrap['low']:%b %d, %Y} to {bootstrap['high']:%b %d, %Y}",
         "Interpretation": "Percentile interval; displayed as uncertainty, not used for refitting."},
    ]
    return pd.DataFrame(rows), {
        "observed_peak": observed_peak, "predicted_peak": predicted_peak,
        "sentinel_start": peak_start, "sentinel_end": peak_end,
        "sentinel_percent": float(sentinel_peak["Positivity_percent"]),
        "in_peak": in_peak,
    }


def make_figures(aligned, lag_df, bootstrap, peak, outdir):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = aligned["Week_midpoint"]
    ax.axvspan(bootstrap["low"], bootstrap["high"], color="#9EC1E6", alpha=0.28,
               label="V5 bootstrap 95% change-point interval")
    ax.axvline(ORIGINAL_CHANGE_POINT, color="#8C1C13", linestyle="--", linewidth=1.6,
               label="Original SEIR change point: May 24")
    ax.plot(x, aligned["Observed_cases_z"], marker="o", linewidth=1.9,
            label="Observed reported cases")
    ax.plot(x, aligned["Predicted_cases_z"], marker="s", linewidth=1.9,
            label="SEIR-predicted incidence")
    ax.plot(x, aligned["Positivity_z"], marker="^", linewidth=2.0,
            label="Sentinel SARS-CoV-2 positivity")
    peak_mid = peak["sentinel_start"] + (peak["sentinel_end"] - peak["sentinel_start"]) / 2
    peak_row = aligned.loc[(aligned["Week_start"] == peak["sentinel_start"]) &
                           (aligned["Week_end"] == peak["sentinel_end"])]
    if not peak_row.empty:
        ax.scatter([peak_mid], [peak_row.iloc[0]["Positivity_z"]], s=85, facecolors="none",
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label="Sentinel positivity peak week")
    ax.set_title("External surveillance comparison of reported cases,\n"
                 "SEIR-predicted incidence, and sentinel SARS-CoV-2 positivity")
    ax.set_xlabel("Week")
    ax.set_ylabel("Standardized value (z-score)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8.5, ncol=2)
    # Observed and predicted lines may overlap because SEIR was fitted to cases;
    # such overlap is expected and is not a plotting or calculation error.
    ax.text(0.01, -0.20,
            "Z-scores use overlapping weeks only. Observed and predicted lines may overlap because the SEIR model was fitted to the observed cases; this is expected.",
            transform=ax.transAxes, fontsize=8, color="#444444")
    fig.autofmt_xdate()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(outdir / "Fig_external_surveillance_comparison.png", dpi=320,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.plot(lag_df["lag_weeks"], lag_df["rho_observed"], marker="o", linewidth=1.9,
            label="Observed cases vs positivity")
    ax.plot(lag_df["lag_weeks"], lag_df["rho_predicted"], marker="s", linewidth=1.9,
            label="SEIR predictions vs positivity")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(LAGS)
    ax.set_xlabel("Lag (weeks; positive = cases later than positivity)")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Exploratory weekly lag analysis")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.text(0.5, -0.19,
            "Exploratory only; the strongest lag is not evidence of a specific biological or reporting delay.",
            ha="center", transform=ax.transAxes, fontsize=8, color="#444444")
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(outdir / "Fig_external_lag_analysis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def strongest_lag(lag_df, column):
    valid = lag_df.dropna(subset=[column])
    if valid.empty:
        return None
    # "Strongest" is defined in advance as greatest absolute rho, with the
    # smaller absolute lag preferred for an exact tie.
    ordered = valid.assign(abs_rho=valid[column].abs(), abs_lag=valid["lag_weeks"].abs())
    return ordered.sort_values(["abs_rho", "abs_lag"], ascending=[False, True]).iloc[0]


def make_summary(
    aligned, correlations, first_difference, lag_df, bootstrap, peak, metadata, exclusions
):
    obs = correlations.iloc[0]
    pred = correlations.iloc[1]
    lag_obs = strongest_lag(lag_df, "rho_observed")
    lag_pred = strongest_lag(lag_df, "rho_predicted")
    exclusion_text = "None" if not exclusions else "\n  - " + "\n  - ".join(exclusions)
    return f"""External validation using China CDC sentinel positivity
========================================================

DATA AND ALIGNMENT
- Number of complete aligned weeks: {len(aligned)}
- Sentinel positivity column: {metadata['positivity_column']!r}
- Week parsing: {metadata['date_source']}
- Positivity input scale: {metadata['positivity_scale']}
- Primary case aggregation for analysis: weekly mean over each stated Excel interval
- Z-scores are computed within the overlapping weeks only (weeks present in all three series), not over each series' full length.
- Excluded rows/weeks: {exclusion_text}

TURNING-POINT COMPARISON
- Original SEIR change point: {ORIGINAL_CHANGE_POINT.date()}
- Observed reported-case peak: {peak['observed_peak'].date()}
- SEIR model-predicted peak: {peak['predicted_peak'].date()}
- Sentinel positivity peak week: {peak['sentinel_start'].date()} to {peak['sentinel_end'].date()} ({peak['sentinel_percent']:.3f}%)
- May 24 lies within sentinel positivity peak week: {str(peak['in_peak']).upper()}
- V5 bootstrap median change point: {bootstrap['median'].date()}
- V5 bootstrap percentile 95% interval: {bootstrap['low'].date()} to {bootstrap['high'].date()}
- Successful V5 replicates used: {bootstrap['n']}

PRIMARY SPEARMAN CORRELATIONS (weekly mean)
- Positivity vs observed cases: rho={obs['rho']:.4f}, p={obs['p_value']:.4g}, n={int(obs['n_weeks'])}
- Positivity vs SEIR-predicted cases: rho={pred['rho']:.4f}, p={pred['p_value']:.4g}, n={int(pred['n_weeks'])}
- Correlations describe temporal association, not causation.

FIRST-DIFFERENCE ROBUSTNESS ANALYSIS
- Differences are calculated between consecutive aligned weeks: delta X_t = X_t - X_(t-1).
- Spearman rho(delta positivity, delta observed cases): rho={first_difference['rho_observed']:.4f}, p={first_difference['p_observed']:.4g}, n={int(first_difference['n_observed'])}
- Spearman rho(delta positivity, delta SEIR-predicted cases): rho={first_difference['rho_predicted']:.4f}, p={first_difference['p_predicted']:.4g}, n={int(first_difference['n_predicted'])}
- This robustness analysis tests whether short-term week-to-week changes remain synchronized after removing the shared level trend; it does not imply causation.

EXPLORATORY LAG ANALYSIS
- Lag convention: positive lag compares positivity at week t with cases at week t+lag.
- Strongest observed-case association: lag={int(lag_obs['lag_weeks']):+d}, rho={lag_obs['rho_observed']:.4f}, p={lag_obs['p_observed']:.4g}, n={int(lag_obs['n_observed'])}
- Strongest predicted-case association: lag={int(lag_pred['lag_weeks']):+d}, rho={lag_pred['rho_predicted']:.4f}, p={lag_pred['p_predicted']:.4g}, n={int(lag_pred['n_predicted'])}
- This lag analysis is exploratory. It does not establish a biological or reporting delay.

EXPECTED PLOT BEHAVIOR
Because the SEIR model was fitted to the observed cases, the observed and predicted case lines in the z-score plot may largely overlap. This is expected and not an error.

INTERPRETATION
The sentinel positivity series was not used in model fitting. Temporal agreement therefore provides additional surveillance-based support for a broad late-May transition, but should not be interpreted as fully independent causal validation or as proof of May 24 exactly.
"""


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    positivity, excel_exclusions, pos_col, date_source, scale = load_positivity(args.positivity)
    daily, prediction_columns = load_predictions(args.predictions)
    bootstrap = load_bootstrap_change_points(args.bootstrap, pd.Timestamp(daily["Date"].min()))
    aligned, alignment_exclusions = align_weekly(positivity, daily)
    exclusions = excel_exclusions + alignment_exclusions

    correlations = correlation_outputs(aligned)
    first_difference = first_difference_outputs(aligned)
    lag_df = lag_outputs(aligned)
    turning_df, peak = make_turning_points(daily, positivity, bootstrap)

    aligned.to_csv(args.output / "external_aligned_weekly.csv", index=False,
                   encoding="utf-8-sig")
    correlations.to_csv(args.output / "external_correlation.csv", index=False,
                        encoding="utf-8-sig")
    lag_df.to_csv(args.output / "external_lag_analysis.csv", index=False,
                  encoding="utf-8-sig")
    turning_df.to_csv(args.output / "turning_point_comparison.csv", index=False,
                      encoding="utf-8-sig")
    make_figures(aligned, lag_df, bootstrap, peak, args.output)

    metadata = {
        "positivity_column": pos_col, "date_source": date_source,
        "positivity_scale": scale, "prediction_columns": prediction_columns,
    }
    summary = make_summary(
        aligned, correlations, first_difference, lag_df, bootstrap, peak, metadata, exclusions
    )
    (args.output / "external_validation_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
