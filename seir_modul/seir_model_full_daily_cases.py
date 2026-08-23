# -*- coding: utf-8 -*-
"""
SEIR v4 - 统一 daily_cases 版本

正式输入文件：
    seir_cleaned.csv

必须包含：
    Date
    daily_cases

说明：
1. 不再使用 Adjusted_Cases。
2. 不再使用 CASE_COL。
3. seir_cleaned.csv 被视为已经完成 CDC 月总数校准后的正式数据。
4. 主模型使用 daily_cases 拟合。
5. 7日移动平均仅用于可视化。
6. 主模型采用数据驱动 change point，不固定 2025-05-26。
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import chi2, ttest_ind

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 0. 配置
# ============================================================

N_POP = 1_408_280_000.0

# 主分析固定生物学参数
MAIN_LATENT_DAYS = 3.0
MAIN_INFECTIOUS_DAYS = 5.0

# q 是 observation scaling factor，不解释为真实报告概率
MAIN_Q = 0.30

# change-point 搜索
MIN_SEGMENT_DAYS = 14
PROFILE_STEP_DAYS = 1

# 优化设置
RNG_SEED = 20260821
MAX_NFEV = 300

# 敏感性分析
CHANGEPOINT_SENS_DATES = [
    "2025-05-20",
    "2025-05-23",
    "2025-05-26",
    "2025-05-29",
    "2025-06-01",
]

Q_SENS_VALUES = [0.10, 0.30, 0.50, 0.70, 0.99]
LATENT_SENS = [2.0, 3.0, 4.0]
INFECTIOUS_SENS = [4.0, 5.0, 7.0]

# CDC通报峰值日期：
# 仅作为 contextual check，不用于选择主模型 change point
CDC_PEAK_DATE = pd.Timestamp("2025-05-26")

# 官方月总数
OFFICIAL_MONTH_TOTALS = {
    "2025-05": 440_662,
    "2025-06": 333_229,
    "2025-07": 226_567,
    "2025-08": 164_625,
    "2025-09": 66_915,
}

# CDC 来源，仅记录 provenance
CDC_SOURCES = {
    "May 2025":
        "https://www.chinacdc.cn/jksj/xgbdyq/202506/t20250605_307356.html",
    "June 2025":
        "https://www.chinacdc.cn/jksj/xgbdyq/202507/t20250703_308176.html",
    "July 2025":
        "https://www.chinacdc.cn/jksj/xgbdyq/202508/t20250806_309202.html",
    "August 2025":
        "https://www.chinacdc.cn/jksj/xgbdyq/202509/t20250905_310229.html",
    "September 2025":
        "https://www.chinacdc.cn/jksj/xgbdyq/202510/t20251011_312903.html",
}

# 唯一正式输入文件
DATA_FILE = Path("seir_cleaned.csv")

# 输出目录
OUTDIR = Path("seir_output_v4")
OUTDIR.mkdir(exist_ok=True)

# 参数边界
BETA_BOUNDS = (0.02, 1.00)
INIT_MULT_BOUNDS = (0.001, 100.0)

rng = np.random.default_rng(RNG_SEED)


# ============================================================
# 1. 数据读取与严格自查
# ============================================================

def load_cleaned_data(path: Path) -> pd.DataFrame:
    """
    正式数据必须满足：
    - 文件名：seir_cleaned.csv
    - 列：Date, daily_cases
    - 日期：2025-05-01 至 2025-09-30
    - 153天连续记录
    - 无重复、无缺失
    - daily_cases 为有限非负数
    - 月总数与 China CDC 官方总数一致
    """

    if not path.exists():
        raise FileNotFoundError(
            f"未找到正式数据文件：{path.resolve()}\n"
            "请把 seir_cleaned.csv 放在脚本同目录。"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    # ---------- 列名 ----------
    required = {"Date", "daily_cases"}
    missing_cols = required - set(df.columns)

    if missing_cols:
        raise ValueError(
            "seir_cleaned.csv 必须包含两列：Date 和 daily_cases。\n"
            f"当前列名：{list(df.columns)}\n"
            f"缺少：{sorted(missing_cols)}"
        )

    # 只取正式建模需要的列
    df = df[["Date", "daily_cases"]].copy()

    # ---------- 类型 ----------
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["daily_cases"] = pd.to_numeric(
        df["daily_cases"], errors="raise"
    ).astype(float)

    # ---------- 重复日期 ----------
    duplicates = df.loc[df["Date"].duplicated(keep=False), "Date"]

    if not duplicates.empty:
        raise ValueError(
            "发现重复日期："
            + ", ".join(d.strftime("%Y-%m-%d") for d in duplicates)
        )

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------- 数值合法性 ----------
    if (~np.isfinite(df["daily_cases"])).any():
        raise ValueError("daily_cases 中存在 NaN 或 Inf。")

    if (df["daily_cases"] < 0).any():
        raise ValueError("daily_cases 中存在负数。")

    # ---------- 研究时间窗口 ----------
    expected_start = pd.Timestamp("2025-05-01")
    expected_end = pd.Timestamp("2025-09-30")
    expected_dates = pd.date_range(expected_start, expected_end, freq="D")

    if df["Date"].iloc[0] != expected_start:
        raise ValueError(
            f"起始日期错误：{df['Date'].iloc[0].date()}，"
            "应为 2025-05-01。"
        )

    if df["Date"].iloc[-1] != expected_end:
        raise ValueError(
            f"结束日期错误：{df['Date'].iloc[-1].date()}，"
            "应为 2025-09-30。"
        )

    if len(df) != 153:
        raise ValueError(
            f"记录数错误：当前 {len(df)} 条，应为 153 条。"
        )

    actual_dates = pd.DatetimeIndex(df["Date"])

    if not actual_dates.equals(expected_dates):
        missing = expected_dates.difference(actual_dates)
        extra = actual_dates.difference(expected_dates)

        raise ValueError(
            "日期序列不连续。\n"
            f"缺失日期：{[d.strftime('%Y-%m-%d') for d in missing]}\n"
            f"额外日期：{[d.strftime('%Y-%m-%d') for d in extra]}"
        )

    # ---------- 月总数核验 ----------
    period = df["Date"].dt.to_period("M").astype(str)
    validation_rows = []

    for month, official_total in OFFICIAL_MONTH_TOTALS.items():

        model_total = float(
            df.loc[period == month, "daily_cases"].sum()
        )

        difference = model_total - official_total
        relative_error_pct = difference / official_total * 100.0

        validation_rows.append(
            {
                "Month": month,
                "daily_cases_sum": model_total,
                "Official_total": official_total,
                "Difference": difference,
                "Relative_error_pct": relative_error_pct,
            }
        )

        # 允许极小浮点误差
        if not np.isclose(
            model_total,
            official_total,
            atol=1.0,
            rtol=1e-8,
        ):
            raise ValueError(
                f"{month} 月总数校验失败：\n"
                f"daily_cases 合计 = {model_total:.3f}\n"
                f"CDC 官方总数 = {official_total}\n"
                "请确认 seir_cleaned.csv 是否为最终校准版本。"
            )

    validation_df = pd.DataFrame(validation_rows)

    validation_df.to_csv(
        OUTDIR / "monthly_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------- 7日移动平均 ----------
    # 只用于画图，不作为主拟合对象
    df["MA7"] = (
        df["daily_cases"]
        .rolling(
            window=7,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    # ---------- 自动识别观测峰值 ----------
    peak_idx = int(df["daily_cases"].idxmax())

    peak_date = pd.Timestamp(
        df.loc[peak_idx, "Date"]
    )

    peak_cases = float(
        df.loc[peak_idx, "daily_cases"]
    )

    # 保存本次真正用于分析的数据快照
    df.to_csv(
        OUTDIR / "analysis_input_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 72)
    print("DATA VALIDATION PASSED")
    print(f"正式输入：{path.resolve()}")
    print(
        f"日期范围："
        f"{df['Date'].iloc[0].date()} -> "
        f"{df['Date'].iloc[-1].date()}"
    )
    print(f"记录数：{len(df)}")
    print(
        f"观测峰值：{peak_date.date()}，"
        f"{peak_cases:.0f} 例/天"
    )

    print("\nCDC 月总数核验：")

    print(
        validation_df[
            [
                "Month",
                "daily_cases_sum",
                "Official_total",
                "Difference",
            ]
        ].to_string(index=False)
    )

    print("=" * 72)

    return df


df = load_cleaned_data(DATA_FILE)

dates = df["Date"]
y = df["daily_cases"].to_numpy(dtype=float)

n = len(y)
c0 = float(y[0])

residual_scale = max(
    float(np.std(y)),
    1.0,
)

t_edges = np.arange(
    n + 1,
    dtype=float,
)

obs_peak_idx = int(np.argmax(y))
OBSERVED_PEAK_DATE = pd.Timestamp(
    dates.iloc[obs_peak_idx]
)
OBSERVED_PEAK_CASES = float(
    y[obs_peak_idx]
)

if OBSERVED_PEAK_DATE != CDC_PEAK_DATE:
    warnings.warn(
        "数据峰值日期与 CDC 通报峰值日期不一致："
        f"数据={OBSERVED_PEAK_DATE.date()}，"
        f"CDC={CDC_PEAK_DATE.date()}。"
    )


# ============================================================
# 2. SEIR simulator
# ============================================================

def simulate_seir(
    beta1: float,
    beta2: float,
    e_mult: float,
    i_mult: float,
    tc_day: int | None,
    sigma: float,
    gamma: float,
    q: float,
    return_states: bool = False,
):

    values = [
        beta1,
        beta2,
        e_mult,
        i_mult,
        sigma,
        gamma,
        q,
    ]

    if (
        not all(np.isfinite(values))
        or min(values) <= 0
    ):
        bad = np.full(n, np.nan)
        return (
            (bad, None)
            if return_states
            else bad
        )

    # nuisance initial states
    E0 = e_mult * c0 / (q * sigma)
    I0 = i_mult * c0 / (q * gamma)

    S0 = N_POP - E0 - I0

    if S0 <= 0:
        bad = np.full(n, np.nan)
        return (
            (bad, None)
            if return_states
            else bad
        )

    def rhs(t, state):

        S, E, I, R, cumulative_onsets = state

        if tc_day is None:
            beta = beta1
        else:
            beta = (
                beta1
                if t < tc_day
                else beta2
            )

        force = beta * S * I / N_POP

        return [
            -force,
            force - sigma * E,
            sigma * E - gamma * I,
            gamma * I,
            sigma * E,
        ]

    sol = solve_ivp(
        rhs,
        (0.0, float(n)),
        [
            S0,
            E0,
            I0,
            0.0,
            0.0,
        ],
        t_eval=t_edges,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )

    if (
        not sol.success
        or sol.y.shape[1] != n + 1
    ):
        bad = np.full(n, np.nan)

        return (
            (bad, None)
            if return_states
            else bad
        )

    # daily reported/model-observed cases
    pred = q * np.diff(
        sol.y[4]
    )

    if return_states:
        return pred, sol.y.T

    return pred


# ============================================================
# 3. Fitting helpers
# ============================================================

def fit_model(
    tc_day: int | None,
    sigma: float,
    gamma: float,
    q: float,
    start: np.ndarray | None = None,
    n_starts: int = 1,
):

    piecewise = tc_day is not None

    if piecewise:

        lower = np.log([
            BETA_BOUNDS[0],
            BETA_BOUNDS[0],
            INIT_MULT_BOUNDS[0],
            INIT_MULT_BOUNDS[0],
        ])

        upper = np.log([
            BETA_BOUNDS[1],
            BETA_BOUNDS[1],
            INIT_MULT_BOUNDS[1],
            INIT_MULT_BOUNDS[1],
        ])

        default = np.log([
            0.28,
            0.17,
            1.0,
            1.0,
        ])

    else:

        lower = np.log([
            BETA_BOUNDS[0],
            INIT_MULT_BOUNDS[0],
            INIT_MULT_BOUNDS[0],
        ])

        upper = np.log([
            BETA_BOUNDS[1],
            INIT_MULT_BOUNDS[1],
            INIT_MULT_BOUNDS[1],
        ])

        default = np.log([
            0.20,
            1.0,
            1.0,
        ])

    starts = []

    if (
        start is not None
        and len(start) == len(default)
    ):
        starts.append(
            np.clip(
                start,
                lower + 1e-9,
                upper - 1e-9,
            )
        )

    if piecewise:

        candidate_starts = [
            [0.28, 0.17, 1.0, 1.0],
            [0.20, 0.12, 0.20, 5.0],
            [0.40, 0.10, 5.0, 0.20],
            [0.30, 0.20, 10.0, 0.10],
            [0.20, 0.08, 0.10, 10.0],
            [0.50, 0.20, 2.0, 2.0],
        ]

    else:

        candidate_starts = [
            [0.20, 1.0, 1.0],
            [0.12, 0.20, 5.0],
            [0.35, 5.0, 0.20],
            [0.18, 10.0, 0.10],
            [0.25, 0.10, 10.0],
            [0.45, 2.0, 2.0],
        ]

    for vals in candidate_starts:

        if len(starts) >= n_starts:
            break

        trial = np.log(vals)

        if not any(
            np.allclose(
                trial,
                existing,
                atol=1e-8,
            )
            for existing in starts
        ):
            starts.append(
                np.clip(
                    trial,
                    lower + 1e-9,
                    upper - 1e-9,
                )
            )

    while len(starts) < n_starts:

        starts.append(
            lower
            + (upper - lower)
            * rng.random(
                len(lower)
            )
        )

    best = None

    for x0 in starts:

        def residual(logp):

            p = np.exp(logp)

            if piecewise:

                pred = simulate_seir(
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    tc_day,
                    sigma,
                    gamma,
                    q,
                )

            else:

                pred = simulate_seir(
                    p[0],
                    p[0],
                    p[1],
                    p[2],
                    None,
                    sigma,
                    gamma,
                    q,
                )

            if not np.all(
                np.isfinite(pred)
            ):
                return np.full(
                    n,
                    1e6,
                )

            return (
                pred - y
            ) / residual_scale

        result = least_squares(
            residual,
            x0,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=MAX_NFEV,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )

        p = np.exp(
            result.x
        )

        if piecewise:

            pred = simulate_seir(
                p[0],
                p[1],
                p[2],
                p[3],
                tc_day,
                sigma,
                gamma,
                q,
            )

        else:

            pred = simulate_seir(
                p[0],
                p[0],
                p[1],
                p[2],
                None,
                sigma,
                gamma,
                q,
            )

        sse = float(
            np.sum(
                (pred - y) ** 2
            )
        )

        if (
            best is None
            or sse < best["sse"]
        ):
            best = {
                "x": result.x.copy(),
                "params": p.copy(),
                "pred": pred.copy(),
                "sse": sse,
                "optimizer_success":
                    bool(result.success),
            }

    return best


def fit_metrics(
    pred: np.ndarray,
    k: int,
) -> dict:

    sse = float(
        np.sum(
            (y - pred) ** 2
        )
    )

    rmse = float(
        np.sqrt(
            sse / n
        )
    )

    sst = float(
        np.sum(
            (y - y.mean()) ** 2
        )
    )

    r2 = float(
        1.0 - sse / sst
    )

    aic = float(
        n * np.log(
            sse / n
        )
        + 2 * k
    )

    bic = float(
        n * np.log(
            sse / n
        )
        + k * np.log(n)
    )

    return {
        "SSE": sse,
        "RMSE": rmse,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
    }


sigma_main = 1.0 / MAIN_LATENT_DAYS
gamma_main = 1.0 / MAIN_INFECTIOUS_DAYS


# ============================================================
# 4. Baseline constant-beta
# ============================================================

print("[1/8] Fitting constant-beta baseline...")

baseline = fit_model(
    None,
    sigma_main,
    gamma_main,
    MAIN_Q,
    n_starts=3,
)

baseline_metrics = fit_metrics(
    baseline["pred"],
    k=3,
)


# ============================================================
# 5. Data-driven change point profile
# ============================================================

print(
    "[2/8] Profiling data-driven change point "
    "(May 26 is NOT fixed)..."
)

profile_rows = []
warm_start = None

candidate_days = range(
    MIN_SEGMENT_DAYS,
    n - MIN_SEGMENT_DAYS,
    PROFILE_STEP_DAYS,
)

for tc_day in candidate_days:

    fit = fit_model(
        tc_day,
        sigma_main,
        gamma_main,
        MAIN_Q,
        start=warm_start,
        n_starts=1,
    )

    warm_start = fit["x"]

    p = fit["params"]

    m = fit_metrics(
        fit["pred"],
        k=5,
    )

    profile_rows.append(
        {
            "tc_day": tc_day,
            "tc_date":
                dates.iloc[0]
                + pd.Timedelta(
                    days=tc_day
                ),
            "beta1": p[0],
            "beta2": p[1],
            "e_mult": p[2],
            "i_mult": p[3],
            "R_before":
                p[0] / gamma_main,
            "R_after":
                p[1] / gamma_main,
            **m,
        }
    )

profile_df = pd.DataFrame(
    profile_rows
)

# 对AIC最好的候选日期精修
top_candidate_days = (
    profile_df
    .nsmallest(
        5,
        "AIC",
    )["tc_day"]
    .astype(int)
    .tolist()
)

refined_fits = {}

for tc_day in top_candidate_days:

    refined = fit_model(
        tc_day,
        sigma_main,
        gamma_main,
        MAIN_Q,
        n_starts=4,
    )

    refined_fits[
        tc_day
    ] = refined

    m = fit_metrics(
        refined["pred"],
        k=5,
    )

    p = refined["params"]

    mask = (
        profile_df["tc_day"]
        == tc_day
    )

    updates = {
        "beta1": p[0],
        "beta2": p[1],
        "e_mult": p[2],
        "i_mult": p[3],
        "R_before":
            p[0] / gamma_main,
        "R_after":
            p[1] / gamma_main,
        **m,
    }

    for key, value in updates.items():

        profile_df.loc[
            mask,
            key,
        ] = value


best_profile_row = profile_df.loc[
    profile_df["AIC"].idxmin()
]

best_tc = int(
    best_profile_row["tc_day"]
)

best_tc_date = pd.Timestamp(
    best_profile_row["tc_date"]
)

piecewise_best = refined_fits.get(
    best_tc
)

if piecewise_best is None:

    piecewise_best = fit_model(
        best_tc,
        sigma_main,
        gamma_main,
        MAIN_Q,
        n_starts=6,
    )

piecewise_metrics = fit_metrics(
    piecewise_best["pred"],
    k=5,
)

bp = piecewise_best[
    "params"
]

print(
    f"Estimated change point: "
    f"{best_tc_date.date()}"
)

print(
    f"beta1={bp[0]:.6f}, "
    f"beta2={bp[1]:.6f}"
)

print(
    f"R_before="
    f"{bp[0]/gamma_main:.3f}, "
    f"R_after="
    f"{bp[1]/gamma_main:.3f}"
)


# ============================================================
# 6. Change-point sensitivity + May26 reference LRT
# ============================================================

print(
    "[3/8] Change-point sensitivity..."
)

cp_sens_rows = []

for text in CHANGEPOINT_SENS_DATES:

    change_date = pd.Timestamp(
        text
    )

    tc_day = int(
        (
            change_date
            - dates.iloc[0]
        ).days
    )

    fit = fit_model(
        tc_day,
        sigma_main,
        gamma_main,
        MAIN_Q,
        start=piecewise_best["x"],
        n_starts=2,
    )

    p = fit["params"]

    m = fit_metrics(
        fit["pred"],
        k=4,
    )

    cp_sens_rows.append(
        {
            "ChangeDate":
                change_date,
            "beta1": p[0],
            "beta2": p[1],
            "R_before":
                p[0] / gamma_main,
            "R_after":
                p[1] / gamma_main,
            **m,
        }
    )


cp_sensitivity_df = pd.DataFrame(
    cp_sens_rows
)

may26_row = cp_sensitivity_df.loc[
    cp_sensitivity_df[
        "ChangeDate"
    ] == CDC_PEAK_DATE
].iloc[0]

fixed_may26_sse = float(
    may26_row["SSE"]
)

reference_lrt = float(
    n
    * np.log(
        baseline_metrics["SSE"]
        / fixed_may26_sse
    )
)

reference_lrt_p = float(
    chi2.sf(
        reference_lrt,
        df=1,
    )
)


# ============================================================
# 7. q sensitivity
# ============================================================

print(
    "[4/8] Observation-scale q sensitivity..."
)

q_rows = []

for q in Q_SENS_VALUES:

    fit = fit_model(
        best_tc,
        sigma_main,
        gamma_main,
        q,
        start=piecewise_best["x"],
        n_starts=3,
    )

    p = fit["params"]

    m = fit_metrics(
        fit["pred"],
        k=4,
    )

    q_rows.append(
        {
            "q": q,
            "beta1": p[0],
            "beta2": p[1],
            "R_before":
                p[0] / gamma_main,
            "R_after":
                p[1] / gamma_main,
            "e_mult": p[2],
            "i_mult": p[3],
            **m,
        }
    )

q_sensitivity_df = pd.DataFrame(
    q_rows
)


# ============================================================
# 8. Biological sensitivity
# ============================================================

print(
    "[5/8] Biological sensitivity..."
)

bio_rows = []

for latent_days in LATENT_SENS:

    for infectious_days in INFECTIOUS_SENS:

        sigma = (
            1.0 / latent_days
        )

        gamma = (
            1.0 / infectious_days
        )

        fit = fit_model(
            best_tc,
            sigma,
            gamma,
            MAIN_Q,
            start=piecewise_best["x"],
            n_starts=3,
        )

        p = fit["params"]

        m = fit_metrics(
            fit["pred"],
            k=4,
        )

        bio_rows.append(
            {
                "Latent_days":
                    latent_days,
                "Infectious_days":
                    infectious_days,
                "beta1": p[0],
                "beta2": p[1],
                "R_before":
                    p[0] / gamma,
                "R_after":
                    p[1] / gamma,
                **m,
            }
        )


bio_sensitivity_df = pd.DataFrame(
    bio_rows
)


# ============================================================
# 9. Relaxed biological bounds diagnostic
# ============================================================

print(
    "[6/8] Relaxed-bound identifiability diagnostic..."
)


def fit_relaxed_biology():

    lower = np.log([
        0.02,
        0.02,
        1.0,
        3.0,
        0.001,
        0.001,
    ])

    upper = np.log([
        1.00,
        1.00,
        7.0,
        10.0,
        100.0,
        100.0,
    ])

    default = np.log([
        bp[0],
        bp[1],
        MAIN_LATENT_DAYS,
        MAIN_INFECTIOUS_DAYS,
        bp[2],
        bp[3],
    ])

    starts = [
        default,
        np.log([
            0.35,
            0.15,
            2.0,
            8.0,
            1.0,
            1.0,
        ]),
        np.log([
            bp[0],
            bp[1],
            6.5,
            9.5,
            bp[2],
            bp[3],
        ]),
    ]

    starts = [
        np.clip(
            x,
            lower + 1e-9,
            upper - 1e-9,
        )
        for x in starts
    ]

    best = None

    for x0 in starts:

        def residual(logp):

            (
                beta1,
                beta2,
                latent_days,
                infectious_days,
                e_mult,
                i_mult,
            ) = np.exp(logp)

            pred = simulate_seir(
                beta1,
                beta2,
                e_mult,
                i_mult,
                best_tc,
                1.0 / latent_days,
                1.0 / infectious_days,
                MAIN_Q,
            )

            if not np.all(
                np.isfinite(pred)
            ):
                return np.full(
                    n,
                    1e6,
                )

            return (
                pred - y
            ) / residual_scale

        result = least_squares(
            residual,
            x0,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=500,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )

        p = np.exp(
            result.x
        )

        pred = simulate_seir(
            p[0],
            p[1],
            p[4],
            p[5],
            best_tc,
            1.0 / p[2],
            1.0 / p[3],
            MAIN_Q,
        )

        sse = float(
            np.sum(
                (pred - y) ** 2
            )
        )

        if (
            best is None
            or sse < best["sse"]
        ):
            best = {
                "params": p,
                "pred": pred,
                "sse": sse,
            }

    return best


relaxed_fit = fit_relaxed_biology()
relaxed_p = relaxed_fit["params"]

relaxed_metrics = fit_metrics(
    relaxed_fit["pred"],
    k=6,
)


def near_bound(
    value: float,
    lower: float,
    upper: float,
    frac: float = 0.02,
) -> bool:

    width = upper - lower

    return (
        (value - lower)
        <= frac * width
        or
        (upper - value)
        <= frac * width
    )


relaxed_latent_near_bound = near_bound(
    relaxed_p[2],
    1.0,
    7.0,
)

relaxed_infectious_near_bound = near_bound(
    relaxed_p[3],
    3.0,
    10.0,
)


# ============================================================
# 10. Residual diagnostics
# ============================================================

print(
    "[7/8] Residual diagnostics..."
)

residuals = (
    y
    - piecewise_best["pred"]
)

residual_df = pd.DataFrame(
    {
        "Date": dates,
        "Observed":
            y,
        "Predicted":
            piecewise_best["pred"],
        "Residual":
            residuals,
        "Weekday":
            dates.dt.day_name(),
    }
)


weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday_df = (
    residual_df
    .groupby("Weekday")
    .agg(
        mean_residual=(
            "Residual",
            "mean",
        ),
        median_residual=(
            "Residual",
            "median",
        ),
        n=(
            "Residual",
            "size",
        ),
    )
    .reindex(
        weekday_order
    )
    .reset_index()
)


weekend_resid = residual_df.loc[
    residual_df[
        "Date"
    ].dt.dayofweek >= 5,
    "Residual",
].to_numpy()


weekday_resid = residual_df.loc[
    residual_df[
        "Date"
    ].dt.dayofweek < 5,
    "Residual",
].to_numpy()


weekend_test = ttest_ind(
    weekend_resid,
    weekday_resid,
    equal_var=False,
)


lag7_corr = float(
    np.corrcoef(
        residuals[:-7],
        residuals[7:],
    )[0, 1]
)


durbin_watson = float(
    np.sum(
        np.diff(
            residuals
        ) ** 2
    )
    /
    np.sum(
        residuals ** 2
    )
)


model_peak_idx = int(
    np.argmax(
        piecewise_best["pred"]
    )
)

model_peak_date = pd.Timestamp(
    dates.iloc[
        model_peak_idx
    ]
)


# ============================================================
# 11. 保存表格
# ============================================================

print(
    "[8/8] Saving outputs and figures..."
)


comparison_df = pd.DataFrame(
    [
        {
            "Model":
                "Constant-beta baseline",
            "k":
                3,
            **baseline_metrics,
        },

        {
            "Model":
                "Piecewise-beta, free change point",
            "k":
                5,
            **piecewise_metrics,
        },

        {
            "Model":
                "Piecewise-beta, fixed May 26 reference",
            "k":
                4,
            "SSE":
                float(
                    may26_row["SSE"]
                ),
            "RMSE":
                float(
                    may26_row["RMSE"]
                ),
            "R2":
                float(
                    may26_row["R2"]
                ),
            "AIC":
                float(
                    may26_row["AIC"]
                ),
            "BIC":
                float(
                    may26_row["BIC"]
                ),
        },
    ]
)


comparison_df.to_csv(
    OUTDIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

profile_df.to_csv(
    OUTDIR / "changepoint_profile.csv",
    index=False,
    encoding="utf-8-sig",
)

cp_sensitivity_df.to_csv(
    OUTDIR / "changepoint_sensitivity.csv",
    index=False,
    encoding="utf-8-sig",
)

q_sensitivity_df.to_csv(
    OUTDIR / "q_sensitivity.csv",
    index=False,
    encoding="utf-8-sig",
)

bio_sensitivity_df.to_csv(
    OUTDIR / "biological_sensitivity.csv",
    index=False,
    encoding="utf-8-sig",
)

residual_df.to_csv(
    OUTDIR / "residuals.csv",
    index=False,
    encoding="utf-8-sig",
)

weekday_df.to_csv(
    OUTDIR / "weekday_residuals.csv",
    index=False,
    encoding="utf-8-sig",
)


relaxed_df = pd.DataFrame(
    [
        {
            "beta1":
                relaxed_p[0],
            "beta2":
                relaxed_p[1],
            "Latent_days":
                relaxed_p[2],
            "Infectious_days":
                relaxed_p[3],
            "e_mult":
                relaxed_p[4],
            "i_mult":
                relaxed_p[5],
            "R_before":
                relaxed_p[0]
                * relaxed_p[3],
            "R_after":
                relaxed_p[1]
                * relaxed_p[3],
            "Latent_near_relaxed_bound":
                relaxed_latent_near_bound,
            "Infectious_near_relaxed_bound":
                relaxed_infectious_near_bound,
            **relaxed_metrics,
        }
    ]
)

relaxed_df.to_csv(
    OUTDIR / "relaxed_bound_diagnostic.csv",
    index=False,
    encoding="utf-8-sig",
)


predictions_df = pd.DataFrame(
    {
        "Date":
            dates,
        "Observed_daily_cases":
            y,
        "Observed_MA7":
            df["MA7"],
        "Baseline_Predicted":
            baseline["pred"],
        "FreeTC_Piecewise_Predicted":
            piecewise_best["pred"],
        "Residual":
            residuals,
    }
)

predictions_df.to_csv(
    OUTDIR / "predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. Figures
# ============================================================

# ---------------- Fig 1 ----------------

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.scatter(
    dates,
    y,
    s=11,
    alpha=0.45,
    label="Daily reported cases",
)

ax.plot(
    dates,
    df["MA7"],
    linewidth=1.8,
    label="7-day moving average",
)

ax.plot(
    dates,
    baseline["pred"],
    linestyle="--",
    linewidth=1.6,
    label=(
        "Constant beta "
        f"(R²={baseline_metrics['R2']:.3f})"
    ),
)

ax.plot(
    dates,
    piecewise_best["pred"],
    linewidth=2.0,
    label=(
        "Free change-point SEIR "
        f"(R²={piecewise_metrics['R2']:.3f})"
    ),
)

ax.axvline(
    best_tc_date,
    linestyle="--",
    linewidth=1.2,
    label=(
        "Estimated change point: "
        f"{best_tc_date:%b %d}"
    ),
)

ax.axvline(
    OBSERVED_PEAK_DATE,
    linestyle=":",
    linewidth=1.2,
    label=(
        "Observed peak: "
        f"{OBSERVED_PEAK_DATE:%b %d}"
    ),
)

ax.set_title(
    "Fig 1. Observed Cases and SEIR Model Fits"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Cases per day"
)

ax.legend()

ax.grid(
    alpha=0.2
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d"
    )
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig1_model_fits.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Fig 2 ----------------

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    profile_df["tc_date"],
    profile_df["AIC"],
    linewidth=1.8,
)

ax.axvline(
    best_tc_date,
    linestyle="--",
    linewidth=1.2,
    label=(
        "AIC minimum: "
        f"{best_tc_date:%b %d}"
    ),
)

ax.axvline(
    OBSERVED_PEAK_DATE,
    linestyle=":",
    linewidth=1.2,
    label=(
        "Observed peak: "
        f"{OBSERVED_PEAK_DATE:%b %d}"
    ),
)

ax.set_title(
    "Fig 2. Data-Driven Change-Point Profile"
)

ax.set_xlabel(
    "Candidate change point"
)

ax.set_ylabel(
    "AIC (lower is better)"
)

ax.legend()

ax.grid(
    alpha=0.2
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d"
    )
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig2_changepoint_profile.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Fig 3 ----------------

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    cp_sensitivity_df["ChangeDate"],
    cp_sensitivity_df["R_before"],
    marker="o",
    label="R before",
)

ax.plot(
    cp_sensitivity_df["ChangeDate"],
    cp_sensitivity_df["R_after"],
    marker="o",
    label="R after",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="R = 1",
)

ax.set_title(
    "Fig 3. Change-Point Sensitivity"
)

ax.set_xlabel(
    "Prespecified change point"
)

ax.set_ylabel(
    "Model-implied reproduction index"
)

ax.legend()

ax.grid(
    alpha=0.2
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d"
    )
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig3_changepoint_sensitivity.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Fig 4 ----------------

bio_plot = bio_sensitivity_df.copy()

bio_plot["Scenario"] = bio_plot.apply(
    lambda r:
        f"L{int(r['Latent_days'])}"
        f"-I{int(r['Infectious_days'])}",
    axis=1,
)

x = np.arange(
    len(bio_plot)
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True,
)

# Fig 4A
axes[0].plot(
    x,
    bio_plot["R_before"],
    marker="o",
    label="R before",
)

axes[0].plot(
    x,
    bio_plot["R_after"],
    marker="o",
    label="R after",
)

axes[0].axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="R = 1",
)

axes[0].set_ylabel(
    "Reproduction index"
)

axes[0].set_title(
    "Fig 4. Biological-Parameter Sensitivity"
)

axes[0].legend()

axes[0].grid(
    alpha=0.2
)

# Fig 4B
axes[1].plot(
    x,
    bio_plot["R2"],
    marker="o",
)

axes[1].axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

axes[1].set_ylabel(
    "R²"
)

axes[1].set_xlabel(
    "Latent / infectious period scenario (days)"
)

axes[1].grid(
    alpha=0.2
)

axes[1].set_xticks(
    x
)

axes[1].set_xticklabels(
    bio_plot["Scenario"]
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig4_biological_sensitivity.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Fig 5 ----------------

is_weekend = (
    dates.dt.dayofweek.to_numpy()
    >= 5
)

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    dates,
    residuals,
    linewidth=1.0,
    label="Residual",
)

ax.scatter(
    dates[is_weekend],
    residuals[is_weekend],
    s=17,
    label="Weekend",
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.2,
)

ax.axvline(
    best_tc_date,
    linestyle=":",
    linewidth=1.2,
    label=(
        "Estimated change point: "
        f"{best_tc_date:%b %d}"
    ),
)

ax.set_title(
    "Fig 5. Residual Diagnostics of the "
    "Free Change-Point SEIR Model"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Residual (Observed - Predicted)"
)

ax.legend()

ax.grid(
    alpha=0.2
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d"
    )
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig5_residuals.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Supplementary Fig S1 ----------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    q_sensitivity_df["q"],
    q_sensitivity_df["R_before"],
    marker="o",
    label="R before",
)

ax.plot(
    q_sensitivity_df["q"],
    q_sensitivity_df["R_after"],
    marker="o",
    label="R after",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="R = 1",
)

ax.set_title(
    "Supplementary Fig S1. "
    "Observation-Scale Sensitivity"
)

ax.set_xlabel(
    "q (observation scaling factor)"
)

ax.set_ylabel(
    "Model-implied reproduction index"
)

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "FigS1_q_sensitivity.png",
    dpi=200,
)

plt.close(fig)


# ---------------- Supplementary Fig S2 ----------------

_, best_states = simulate_seir(
    bp[0],
    bp[1],
    bp[2],
    bp[3],
    best_tc,
    sigma_main,
    gamma_main,
    MAIN_Q,
    return_states=True,
)

state_dates = pd.date_range(
    dates.iloc[0],
    periods=n + 1,
    freq="D",
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.plot(
    state_dates,
    best_states[:, 1],
    label="E",
)

ax.plot(
    state_dates,
    best_states[:, 2],
    label="I",
)

ax.plot(
    state_dates,
    best_states[:, 3],
    label="R",
)

ax.set_yscale(
    "log"
)

ax.set_title(
    "Supplementary Fig S2. "
    "SEIR Compartments "
    "(Scale Depends on q)"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Individuals (log scale)"
)

ax.legend()

ax.grid(
    alpha=0.2
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d"
    )
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    OUTDIR / "FigS2_compartments.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 13. Summary
# ============================================================

robust_cp = bool(
    (
        (
            cp_sensitivity_df["R_before"]
            > 1
        )
        &
        (
            cp_sensitivity_df["R_after"]
            < 1
        )
    ).all()
)

robust_q = bool(
    (
        (
            q_sensitivity_df["R_before"]
            > 1
        )
        &
        (
            q_sensitivity_df["R_after"]
            < 1
        )
    ).all()
)

bio_direction_mask = (
    (
        bio_sensitivity_df["R_before"]
        > 1
    )
    &
    (
        bio_sensitivity_df["R_after"]
        < 1
    )
)

robust_bio = bool(
    bio_direction_mask.all()
)

bio_robust_count = int(
    bio_direction_mask.sum()
)


summary = f"""
SEIR v4 unified daily_cases analysis
====================================

DATA
- {n} daily observations
- {dates.iloc[0].date()} to {dates.iloc[-1].date()}
- Fit variable: daily_cases
- Input file: seir_cleaned.csv
- 7-day moving average is visualization only.

OBSERVED DATA
- Observed peak:
  {OBSERVED_PEAK_DATE.date()}
  ({OBSERVED_PEAK_CASES:.0f} cases/day)

PRIMARY MODEL DESIGN
- Baseline:
  beta + E0 nuisance multiplier + I0 nuisance multiplier
  k=3

- Free-change-point model:
  beta1 + beta2 + E0 nuisance multiplier
  + I0 nuisance multiplier + tc
  k=5

- Latent period:
  {MAIN_LATENT_DAYS:g} days

- Infectious period:
  {MAIN_INFECTIOUS_DAYS:g} days

- q:
  {MAIN_Q:.2f}

q is an observation scaling factor,
NOT a directly estimated infection reporting probability.

DATA-DRIVEN CHANGE POINT
- Estimated tc:
  {best_tc_date.date()}

- beta1:
  {bp[0]:.6f}

- beta2:
  {bp[1]:.6f}

- R_before:
  {bp[0]/gamma_main:.4f}

- R_after:
  {bp[1]/gamma_main:.4f}

MODEL COMPARISON
- Baseline:
  R2={baseline_metrics['R2']:.4f}
  RMSE={baseline_metrics['RMSE']:.2f}
  AIC={baseline_metrics['AIC']:.2f}
  BIC={baseline_metrics['BIC']:.2f}

- Free change-point:
  R2={piecewise_metrics['R2']:.4f}
  RMSE={piecewise_metrics['RMSE']:.2f}
  AIC={piecewise_metrics['AIC']:.2f}
  BIC={piecewise_metrics['BIC']:.2f}

STATISTICAL TESTING
- Primary free-tc comparison:
  use AIC/BIC rather than ordinary chi-square LRT.

- Fixed May 26 reference LRT only:
  statistic={reference_lrt:.3f}
  df=1
  p={reference_lrt_p:.3e}

ROBUSTNESS
- Change-point sensitivity:
  all R_before>1 and R_after<1 = {robust_cp}

- q sensitivity:
  all R_before>1 and R_after<1 = {robust_q}

- Biological sensitivity:
  {bio_robust_count}/9 scenarios satisfy
  R_before>1 and R_after<1

- Biological sensitivity R2 range:
  {bio_sensitivity_df['R2'].min():.3f}
  to
  {bio_sensitivity_df['R2'].max():.3f}

PEAK CONSISTENCY
- Observed peak:
  {OBSERVED_PEAK_DATE.date()}

- Model peak:
  {model_peak_date.date()}

The observed peak is NOT used to choose tc.

RELAXED-BOUND DIAGNOSTIC
- latent:
  {relaxed_p[2]:.3f} days

- infectious:
  {relaxed_p[3]:.3f} days

- latent near boundary:
  {relaxed_latent_near_bound}

- infectious near boundary:
  {relaxed_infectious_near_bound}

RESIDUAL DIAGNOSTICS
- Mean weekend residual:
  {weekend_resid.mean():.1f}

- Mean weekday residual:
  {weekday_resid.mean():.1f}

- Welch p:
  {weekend_test.pvalue:.3e}

- Lag-7 residual correlation:
  {lag7_corr:.3f}

- Durbin-Watson:
  {durbin_watson:.3f}

CORE CLAIM
The preferred model supports a transition
from a growth regime (R>1)
to a decline regime (R<1).

Do not interpret q, latent period,
infectious period, E0, or I0
as precisely identified biological truths.
""".strip()


(
    OUTDIR / "analysis_summary.txt"
).write_text(
    summary,
    encoding="utf-8",
)


source_text = (
    "China CDC sources:\n"
    + "\n".join(
        f"- {k}: {v}"
        for k, v
        in CDC_SOURCES.items()
    )
)

(
    OUTDIR / "cdc_sources.txt"
).write_text(
    source_text,
    encoding="utf-8",
)


print(
    "\n"
    + summary
)

print(
    "\n输出目录：",
    OUTDIR.resolve(),
)

print(
    "\n完成。"
)
