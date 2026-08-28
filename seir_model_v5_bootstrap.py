# -*- coding: utf-8 -*-
"""
SEIR V5: V4 point estimation + 7-day residual block bootstrap.

The V4 source and seir_output_v4 are never modified.  By default this script
runs B=20 (debug); use ``--bootstrap-replicates 500`` for the final analysis.
Every bootstrap replicate repeats the full free-change-point search.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


N_POP = 1_408_280_000.0
MAIN_LATENT_DAYS = 3.0
MAIN_INFECTIOUS_DAYS = 5.0
MAIN_Q = 0.30
MIN_SEGMENT_DAYS = 14
PROFILE_STEP_DAYS = 1
MAX_NFEV = 300
BETA_BOUNDS = (0.02, 1.00)
INIT_MULT_BOUNDS = (0.001, 100.0)
RNG_SEED = 20260821
BLOCK_LENGTH = 7

OFFICIAL_MONTH_TOTALS = {
    "2025-05": 440_662,
    "2025-06": 333_229,
    "2025-07": 226_567,
    "2025-08": 164_625,
    "2025-09": 66_915,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=here / "seir_cleaned.csv",
        help="Path to seir_cleaned.csv (default: beside this script).",
    )
    parser.add_argument(
        "--output", type=Path, default=here / "seir_output_v5",
        help="V5 output directory (must not be seir_output_v4).",
    )
    parser.add_argument(
        "--bootstrap-replicates", "-B", type=int, default=500,
        help="Number of replicates; use 20 for debug and 500 for final.",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def load_cleaned_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path.resolve()}\n"
            "Place seir_cleaned.csv beside this script or pass --data PATH."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Date", "daily_cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[["Date", "daily_cases"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["daily_cases"] = pd.to_numeric(df["daily_cases"], errors="raise").astype(float)
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")
    df = df.sort_values("Date").reset_index(drop=True)
    expected = pd.date_range("2025-05-01", "2025-09-30", freq="D")
    if not pd.DatetimeIndex(df["Date"]).equals(expected):
        raise ValueError("Expected 153 consecutive dates from 2025-05-01 to 2025-09-30.")
    values = df["daily_cases"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("daily_cases must contain finite non-negative values.")
    periods = df["Date"].dt.to_period("M").astype(str)
    for month, official in OFFICIAL_MONTH_TOTALS.items():
        total = float(df.loc[periods == month, "daily_cases"].sum())
        if not np.isclose(total, official, atol=1.0, rtol=1e-8):
            raise ValueError(f"Monthly validation failed for {month}: {total} != {official}")
    return df


class V4Fitter:
    """V4 fitting logic, parameterized by the observation vector."""

    def __init__(self, dates: pd.Series, y_obs: np.ndarray, rng: np.random.Generator):
        self.dates = dates.reset_index(drop=True)
        self.y = np.asarray(y_obs, dtype=float)
        self.n = len(self.y)
        self.c0 = float(self.y[0])
        self.residual_scale = max(float(np.std(self.y)), 1.0)
        self.t_edges = np.arange(self.n + 1, dtype=float)
        self.rng = rng
        self.sigma = 1.0 / MAIN_LATENT_DAYS
        self.gamma = 1.0 / MAIN_INFECTIOUS_DAYS

    def simulate_seir(self, beta1, beta2, e_mult, i_mult, tc_day):
        vals = [beta1, beta2, e_mult, i_mult, self.sigma, self.gamma, MAIN_Q]
        if not all(np.isfinite(vals)) or min(vals) <= 0:
            return np.full(self.n, np.nan)
        e0 = e_mult * self.c0 / (MAIN_Q * self.sigma)
        i0 = i_mult * self.c0 / (MAIN_Q * self.gamma)
        s0 = N_POP - e0 - i0
        if s0 <= 0:
            return np.full(self.n, np.nan)

        def rhs(t, state):
            s, e, i, r, cumulative_onsets = state
            beta = beta1 if tc_day is None or t < tc_day else beta2
            force = beta * s * i / N_POP
            return [-force, force - self.sigma * e, self.sigma * e - self.gamma * i,
                    self.gamma * i, self.sigma * e]

        sol = solve_ivp(
            rhs, (0.0, float(self.n)), [s0, e0, i0, 0.0, 0.0],
            t_eval=self.t_edges, method="LSODA", rtol=1e-6, atol=1e-8,
        )
        if not sol.success or sol.y.shape[1] != self.n + 1:
            return np.full(self.n, np.nan)
        return MAIN_Q * np.diff(sol.y[4])

    def fit_model(self, tc_day, start=None, n_starts=1):
        piecewise = tc_day is not None
        if piecewise:
            lower = np.log([BETA_BOUNDS[0], BETA_BOUNDS[0],
                            INIT_MULT_BOUNDS[0], INIT_MULT_BOUNDS[0]])
            upper = np.log([BETA_BOUNDS[1], BETA_BOUNDS[1],
                            INIT_MULT_BOUNDS[1], INIT_MULT_BOUNDS[1]])
            candidates = [[0.28, 0.17, 1.0, 1.0], [0.20, 0.12, 0.20, 5.0],
                          [0.40, 0.10, 5.0, 0.20], [0.30, 0.20, 10.0, 0.10],
                          [0.20, 0.08, 0.10, 10.0], [0.50, 0.20, 2.0, 2.0]]
        else:
            lower = np.log([BETA_BOUNDS[0], INIT_MULT_BOUNDS[0], INIT_MULT_BOUNDS[0]])
            upper = np.log([BETA_BOUNDS[1], INIT_MULT_BOUNDS[1], INIT_MULT_BOUNDS[1]])
            candidates = [[0.20, 1.0, 1.0], [0.12, 0.20, 5.0],
                          [0.35, 5.0, 0.20], [0.18, 10.0, 0.10],
                          [0.25, 0.10, 10.0], [0.45, 2.0, 2.0]]
        starts = []
        if start is not None and len(start) == len(lower):
            starts.append(np.clip(start, lower + 1e-9, upper - 1e-9))
        for vals in candidates:
            if len(starts) >= n_starts:
                break
            trial = np.log(vals)
            if not any(np.allclose(trial, old, atol=1e-8) for old in starts):
                starts.append(np.clip(trial, lower + 1e-9, upper - 1e-9))
        while len(starts) < n_starts:
            starts.append(lower + (upper - lower) * self.rng.random(len(lower)))

        best = None
        for x0 in starts:
            def residual(logp):
                p = np.exp(logp)
                pred = (self.simulate_seir(p[0], p[1], p[2], p[3], tc_day)
                        if piecewise else
                        self.simulate_seir(p[0], p[0], p[1], p[2], None))
                if not np.all(np.isfinite(pred)):
                    return np.full(self.n, 1e6)
                return (pred - self.y) / self.residual_scale

            result = least_squares(
                residual, x0, bounds=(lower, upper), max_nfev=MAX_NFEV,
                xtol=1e-8, ftol=1e-8, gtol=1e-8,
            )
            p = np.exp(result.x)
            pred = (self.simulate_seir(p[0], p[1], p[2], p[3], tc_day)
                    if piecewise else
                    self.simulate_seir(p[0], p[0], p[1], p[2], None))
            sse = float(np.sum((pred - self.y) ** 2))
            if np.isfinite(sse) and (best is None or sse < best["sse"]):
                best = {"x": result.x.copy(), "params": p.copy(), "pred": pred.copy(),
                        "sse": sse, "optimizer_success": bool(result.success)}
        if best is None:
            raise RuntimeError("All optimizer starts failed.")
        return best

    def metrics(self, pred, k=5):
        sse = float(np.sum((self.y - pred) ** 2))
        rmse = float(np.sqrt(sse / self.n))
        sst = float(np.sum((self.y - self.y.mean()) ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
        safe_sse = max(sse, np.finfo(float).tiny)
        aic = float(self.n * np.log(safe_sse / self.n) + 2 * k)
        bic = float(self.n * np.log(safe_sse / self.n) + k * np.log(self.n))
        return {"SSE": sse, "RMSE": rmse, "R2": r2, "AIC": aic, "BIC": bic}

    def fit_free_change_point(self):
        rows, warm_start = [], None
        candidates = range(MIN_SEGMENT_DAYS, self.n - MIN_SEGMENT_DAYS, PROFILE_STEP_DAYS)
        for tc_day in candidates:
            fit = self.fit_model(tc_day, start=warm_start, n_starts=1)
            warm_start = fit["x"]
            p = fit["params"]
            rows.append({"tc_day": tc_day, "beta1": p[0], "beta2": p[1],
                         "e_mult": p[2], "i_mult": p[3],
                         "R_before": p[0] / self.gamma, "R_after": p[1] / self.gamma,
                         **self.metrics(fit["pred"], k=5)})
        profile = pd.DataFrame(rows)
        refined_fits = {}
        for tc_day in profile.nsmallest(5, "AIC")["tc_day"].astype(int):
            fit = self.fit_model(tc_day, n_starts=4)
            refined_fits[tc_day] = fit
            p = fit["params"]
            updates = {"beta1": p[0], "beta2": p[1], "e_mult": p[2], "i_mult": p[3],
                       "R_before": p[0] / self.gamma, "R_after": p[1] / self.gamma,
                       **self.metrics(fit["pred"], k=5)}
            mask = profile["tc_day"] == tc_day
            for key, value in updates.items():
                profile.loc[mask, key] = value
        best_row = profile.loc[profile["AIC"].idxmin()]
        tc_day = int(best_row["tc_day"])
        fit = refined_fits.get(tc_day) or self.fit_model(tc_day, n_starts=6)
        p = fit["params"]
        return {
            "change_index": tc_day,
            "change_date": pd.Timestamp(self.dates.iloc[0]) + pd.Timedelta(days=tc_day),
            "beta1": float(p[0]), "beta2": float(p[1]),
            "e_mult": float(p[2]), "i_mult": float(p[3]),
            "R_before": float(p[0] / self.gamma), "R_after": float(p[1] / self.gamma),
            "prediction": fit["pred"], "optimizer_success": fit["optimizer_success"],
            **self.metrics(fit["pred"], k=5),
        }


def calendar_aligned_block_resample(residuals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample aligned consecutive 7-day residual blocks, then truncate to n days."""
    n = len(residuals)
    starts = np.arange(0, n - BLOCK_LENGTH + 1, BLOCK_LENGTH)
    if len(starts) == 0:
        raise ValueError("Series is shorter than the block length.")
    needed = int(np.ceil(n / BLOCK_LENGTH))
    chosen = rng.choice(starts, size=needed, replace=True)
    sampled = np.concatenate([residuals[s:s + BLOCK_LENGTH] for s in chosen])
    return sampled[:n]


def percentile(values):
    arr = np.asarray(values, dtype=float)
    return tuple(np.percentile(arr, [2.5, 50.0, 97.5]))


def make_outputs(outdir, dates, original, results, requested, block_length):
    successful = results.loc[results["success"]].copy()
    n_success = len(successful)
    rate = n_success / requested if requested else np.nan
    if n_success == 0:
        raise RuntimeError("No successful bootstrap replicates; inspect bootstrap_results.csv.")

    summary_rows = []
    for parameter in ["beta1", "beta2", "R_before", "R_after"]:
        low, median, high = percentile(successful[parameter])
        summary_rows.append({"parameter": parameter, "original_estimate": original[parameter],
                             "bootstrap_median": median, "ci_2.5_pct": low,
                             "ci_97.5_pct": high, "n_successful": n_success})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    frequency = (successful.groupby(["change_index", "change_date"]).size()
                 .rename("frequency").reset_index().sort_values("change_index"))
    frequency["proportion"] = frequency["frequency"] / n_success
    frequency.to_csv(outdir / "changepoint_frequency.csv", index=False, encoding="utf-8-sig")

    cp_low, cp_median, cp_high = percentile(successful["change_index"])
    start_date = pd.Timestamp(dates.iloc[0])
    cp_low_date = start_date + pd.Timedelta(days=int(np.floor(cp_low)))
    cp_median_date = start_date + pd.Timedelta(days=int(np.rint(cp_median)))
    cp_high_date = start_date + pd.Timedelta(days=int(np.ceil(cp_high)))
    p_before = float((successful["R_before"] > 1).mean())
    p_after = float((successful["R_after"] < 1).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col, label, color in zip(
        axes, ["R_before", "R_after"], ["R before", "R after"], ["#2878B5", "#D1495B"]
    ):
        ax.hist(successful[col], bins="auto", color=color, alpha=0.78, edgecolor="white")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="R = 1")
        ax.axvline(original[col], color="#555555", linestyle=":", linewidth=1.5,
                   label="Original estimate")
        ax.set_xlabel(label)
        ax.set_ylabel("Bootstrap replicates")
        ax.legend()
        ax.grid(alpha=0.18)
    fig.suptitle("Bootstrap distributions of model-implied reproduction indices")
    fig.tight_layout()
    fig.savefig(outdir / "FigS4_bootstrap_R_distributions.png", dpi=200)
    plt.close(fig)

    frequency["change_date"] = pd.to_datetime(frequency["change_date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(frequency["change_date"], frequency["frequency"], width=0.85,
           color="#4C956C", alpha=0.85)
    ax.axvline(original["change_date"], color="#C43C39", linestyle="--", linewidth=1.6,
               label=f"Original: {original['change_date']:%b %d}")
    ax.set_title("Bootstrap frequency of the estimated free change point")
    ax.set_xlabel("Estimated change point")
    ax.set_ylabel("Bootstrap replicates")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "FigS5_bootstrap_changepoint_frequency.png", dpi=200)
    plt.close(fig)

    ci = {row["parameter"]: row for row in summary_rows}
    text = f"""SEIR V5: 7-day block bootstrap uncertainty
================================================

DESIGN
- Original point-estimate logic: unchanged from V4
- Residual bootstrap: calendar-aligned consecutive {block_length}-day blocks
- Every replicate re-searches the free change point
- Requested replicates: {requested}
- Successful replicates: {n_success}
- Failed replicates: {requested - n_success}
- Success rate: {rate:.1%}

ORIGINAL-DATA POINT ESTIMATES (not replaced by bootstrap medians)
- Change point: {original['change_date'].date()}
- beta1: {original['beta1']:.6f}
- beta2: {original['beta2']:.6f}
- R_before: {original['R_before']:.4f}
- R_after: {original['R_after']:.4f}

PERCENTILE 95% BOOTSTRAP INTERVALS
- beta1: {original['beta1']:.6f} (95% CI {ci['beta1']['ci_2.5_pct']:.6f} to {ci['beta1']['ci_97.5_pct']:.6f})
- beta2: {original['beta2']:.6f} (95% CI {ci['beta2']['ci_2.5_pct']:.6f} to {ci['beta2']['ci_97.5_pct']:.6f})
- R_before: {original['R_before']:.4f} (95% CI {ci['R_before']['ci_2.5_pct']:.4f} to {ci['R_before']['ci_97.5_pct']:.4f})
- R_after: {original['R_after']:.4f} (95% CI {ci['R_after']['ci_2.5_pct']:.4f} to {ci['R_after']['ci_97.5_pct']:.4f})

CHANGE-POINT UNCERTAINTY
- Original change point: {original['change_date'].date()}
- Bootstrap median date: {cp_median_date.date()}
- Percentile 95% interval: {cp_low_date.date()} to {cp_high_date.date()}

THRESHOLD PROBABILITIES
- P(R_before > 1): {p_before:.4f}
- P(R_after < 1): {p_after:.4f}

NOTE
B=20 is a pipeline/debug run and should not be reported as final inference.
For the planned final run, use --bootstrap-replicates 500.
"""
    (outdir / "V5_summary.txt").write_text(text, encoding="utf-8")
    return text


def main():
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive.")
    if args.output.name.lower() == "seir_output_v4":
        raise ValueError("Refusing to write into seir_output_v4.")
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_cleaned_data(args.data)
    dates = df["Date"]
    y = df["daily_cases"].to_numpy(dtype=float)
    master_rng = np.random.default_rng(args.seed)

    print("Fitting original data with the unchanged V4 point-estimate logic...")
    original = V4Fitter(dates, y, master_rng).fit_free_change_point()
    original_row = {k: v for k, v in original.items() if k != "prediction"}
    pd.DataFrame([original_row]).to_csv(
        args.output / "v4_reference_results.csv", index=False, encoding="utf-8-sig"
    )
    fitted = original["prediction"]
    residuals = y - fitted

    rows = []
    print(f"Running {args.bootstrap_replicates} bootstrap replicates...")
    for replicate in range(1, args.bootstrap_replicates + 1):
        replicate_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        replicate_rng = np.random.default_rng(replicate_seed)
        try:
            sampled_residuals = calendar_aligned_block_resample(residuals, replicate_rng)
            y_star = np.maximum(0.0, fitted + sampled_residuals)
            result = V4Fitter(dates, y_star, replicate_rng).fit_free_change_point()
            if not result["optimizer_success"]:
                raise RuntimeError("Final optimizer did not report convergence.")
            required = [result[k] for k in ["beta1", "beta2", "R_before", "R_after", "AIC"]]
            if not np.isfinite(required).all():
                raise RuntimeError("Final fit contains non-finite estimates.")
            rows.append({"replicate": replicate, "seed": replicate_seed, "success": True,
                         "error": "", **{k: v for k, v in result.items() if k != "prediction"}})
            print(f"  {replicate}/{args.bootstrap_replicates}: {result['change_date'].date()} OK")
        except Exception as exc:
            rows.append({"replicate": replicate, "seed": replicate_seed, "success": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {replicate}/{args.bootstrap_replicates}: FAILED ({type(exc).__name__})")

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "bootstrap_results.csv", index=False, encoding="utf-8-sig")
    try:
        summary = make_outputs(
            args.output, dates, original, results, args.bootstrap_replicates, BLOCK_LENGTH
        )
    except Exception:
        (args.output / "V5_summary.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print("\n" + summary)
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
