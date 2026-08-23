import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. 路径与固定假设
# ============================================================

N = 1_408_280_000.0

DATA_FILE = Path("seir_cleaned.csv")
RESULT_DIR = Path("seir_output_v4")

MAIN_LATENT_DAYS = 3.0
MAIN_INFECTIOUS_DAYS = 5.0
MAIN_Q = 0.30

SIGMA = 1.0 / MAIN_LATENT_DAYS
GAMMA = 1.0 / MAIN_INFECTIOUS_DAYS

OUTDIR = Path("final_figures_v4")
OUTDIR.mkdir(exist_ok=True)


# ============================================================
# 2. 读取正式数据
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"未找到 {DATA_FILE.resolve()}。"
    )

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# 兼容 date / Date
colmap = {c.strip().lower(): c for c in df.columns}

if "date" not in colmap:
    raise ValueError(
        f"没有找到日期列。当前列名：{list(df.columns)}"
    )

if "daily_cases" not in colmap:
    raise ValueError(
        f"没有找到 daily_cases。当前列名：{list(df.columns)}"
    )

df = df.rename(
    columns={
        colmap["date"]: "Date",
        colmap["daily_cases"]: "daily_cases",
    }
)

df = df[["Date", "daily_cases"]].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["daily_cases"] = pd.to_numeric(
    df["daily_cases"], errors="raise"
).astype(float)

df = df.sort_values("Date").reset_index(drop=True)

dates = df["Date"]
y = df["daily_cases"].to_numpy(dtype=float)

n = len(df)
c0 = float(y[0])

if n != 153:
    raise ValueError(
        f"记录数为 {n}，预期应为 153。"
    )

df["MA7"] = (
    df["daily_cases"]
    .rolling(7, center=True, min_periods=1)
    .mean()
)

observed_peak_idx = int(np.argmax(y))
observed_peak_date = pd.Timestamp(
    dates.iloc[observed_peak_idx]
)
observed_peak_cases = float(
    y[observed_peak_idx]
)


# ============================================================
# 3. 读取主模型输出
# ============================================================

required_files = {
    "comparison": RESULT_DIR / "model_comparison.csv",
    "profile": RESULT_DIR / "changepoint_profile.csv",
    "cp_sensitivity": RESULT_DIR / "changepoint_sensitivity.csv",
    "bio_sensitivity": RESULT_DIR / "biological_sensitivity.csv",
    "q_sensitivity": RESULT_DIR / "q_sensitivity.csv",
    "predictions": RESULT_DIR / "predictions.csv",
    "residuals": RESULT_DIR / "residuals.csv",
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"缺少主模型输出：{path}\n"
            "请先运行 seir_model_full_daily_cases.py。"
        )

comparison_df = pd.read_csv(required_files["comparison"])
profile_df = pd.read_csv(required_files["profile"])
cp_sens_df = pd.read_csv(required_files["cp_sensitivity"])
bio_df = pd.read_csv(required_files["bio_sensitivity"])
q_df = pd.read_csv(required_files["q_sensitivity"])
pred_df = pd.read_csv(required_files["predictions"])
resid_df = pd.read_csv(required_files["residuals"])

profile_df["tc_date"] = pd.to_datetime(profile_df["tc_date"])
cp_sens_df["ChangeDate"] = pd.to_datetime(cp_sens_df["ChangeDate"])
pred_df["Date"] = pd.to_datetime(pred_df["Date"])
resid_df["Date"] = pd.to_datetime(resid_df["Date"])

# 最优 change point：直接从 AIC profile 找
best_row = profile_df.loc[
    profile_df["AIC"].idxmin()
].copy()

BEST_TC_DATE = pd.Timestamp(best_row["tc_date"])
BEST_TC_DAY = int(best_row["tc_day"])

BETA1 = float(best_row["beta1"])
BETA2 = float(best_row["beta2"])
E_MULT = float(best_row["e_mult"])
I_MULT = float(best_row["i_mult"])

R_BEFORE = BETA1 / GAMMA
R_AFTER = BETA2 / GAMMA

# 模型比较结果
base_row = comparison_df.loc[
    comparison_df["Model"].str.contains(
        "Constant-beta", case=False, regex=False
    )
].iloc[0]

piece_row = comparison_df.loc[
    comparison_df["Model"].str.contains(
        "free change point", case=False, regex=False
    )
].iloc[0]

BASE_R2 = float(base_row["R2"])
PIECE_R2 = float(piece_row["R2"])

# 预测结果
baseline_pred = pred_df["Baseline_Predicted"].to_numpy(float)
piecewise_pred = pred_df["FreeTC_Piecewise_Predicted"].to_numpy(float)
residuals = pred_df["Residual"].to_numpy(float)

if len(baseline_pred) != n or len(piecewise_pred) != n:
    raise ValueError("主模型 predictions.csv 长度与 seir_cleaned.csv 不一致。")


print("=" * 72)
print("独立出图脚本 V4")
print(f"数据：{dates.iloc[0].date()} -> {dates.iloc[-1].date()} ({n} 天)")
print(f"观测峰值：{observed_peak_date.date()}，{observed_peak_cases:.0f} 例/天")
print(f"AIC 最优变点：{BEST_TC_DATE.date()}")
print(f"beta1={BETA1:.6f}, beta2={BETA2:.6f}")
print(f"R_before={R_BEFORE:.4f}, R_after={R_AFTER:.4f}")
print(f"Baseline R²={BASE_R2:.4f}")
print(f"Piecewise R²={PIECE_R2:.4f}")
print("=" * 72)


# ============================================================
# 4. 仅为舱室图和反事实图重建状态轨迹
#    注意：不是重新拟合，只是用已经拟合好的参数做 ODE 模拟
# ============================================================

t_edges = np.arange(n + 1, dtype=float)

def simulate_from_final_params(
    beta1,
    beta2,
    e_mult,
    i_mult,
    tc_day,
    sigma,
    gamma,
    q,
):
    E0 = e_mult * c0 / (q * sigma)
    I0 = i_mult * c0 / (q * gamma)
    S0 = N - E0 - I0

    if S0 <= 0:
        raise ValueError("初始条件异常：S0 <= 0")

    def rhs(t, state):
        S, E, I, R, Cum = state

        if tc_day is None:
            beta = beta1
        else:
            beta = beta1 if t < tc_day else beta2

        force = beta * S * I / N

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
        [S0, E0, I0, 0.0, 0.0],
        t_eval=t_edges,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE 求解失败：{sol.message}"
        )

    pred = q * np.diff(sol.y[4])

    return sol, pred


final_sol, reconstructed_pred = simulate_from_final_params(
    BETA1,
    BETA2,
    E_MULT,
    I_MULT,
    BEST_TC_DAY,
    SIGMA,
    GAMMA,
    MAIN_Q,
)

# 与主模型预测做一致性检查
max_abs_diff = float(
    np.max(np.abs(reconstructed_pred - piecewise_pred))
)

print(
    f"状态轨迹重建与主模型预测最大绝对差："
    f"{max_abs_diff:.6f}"
)

if max_abs_diff > 5.0:
    print(
        "警告：重建预测与主模型 predictions.csv 有一定差异；"
        "Fig1/Fig5 仍直接使用主模型保存的预测，"
        "因此不会影响正式拟合图。"
    )

# 反事实：保持 beta1，不发生拟合得到的传播率下降
counter_sol, counter_pred = simulate_from_final_params(
    BETA1,
    BETA1,
    E_MULT,
    I_MULT,
    None,
    SIGMA,
    GAMMA,
    MAIN_Q,
)

state_dates = pd.date_range(
    dates.iloc[0],
    periods=n + 1,
    freq="D",
)


# ============================================================
# 5. Fig 1 — 最终拟合图
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    dates,
    y,
    s=12,
    alpha=0.45,
    label="Observed daily cases",
)

ax.plot(
    dates,
    df["MA7"],
    linewidth=1.6,
    label="7-day moving average",
)

ax.plot(
    dates,
    baseline_pred,
    "--",
    linewidth=1.6,
    label=f"Constant-beta SEIR (R²={BASE_R2:.3f})",
)

ax.plot(
    dates,
    piecewise_pred,
    linewidth=2.0,
    label=f"Free change-point SEIR (R²={PIECE_R2:.3f})",
)

ax.axvline(
    BEST_TC_DATE,
    linestyle="--",
    linewidth=1.3,
    label=f"Estimated change point: {BEST_TC_DATE:%b %d}",
)

ax.axvline(
    observed_peak_date,
    linestyle=":",
    linewidth=1.3,
    label=f"Observed peak: {observed_peak_date:%b %d}",
)

ax.set_title(
    "Observed vs. Model-Predicted Daily Cases"
)
ax.set_xlabel("Date")
ax.set_ylabel("Daily reported cases")
ax.legend()
ax.grid(alpha=0.2)
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig1_final_model_fit.png",
    dpi=220,
)

plt.close(fig)

print("已保存 Fig1_final_model_fit.png")


# ============================================================
# 6. Fig 2 — SEIR compartments
#    建议作为 supplementary figure
# ============================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 8),
    sharex=True,
)

axes[0].plot(
    state_dates,
    final_sol.y[1],
    label="Exposed E(t)",
)
axes[0].plot(
    state_dates,
    final_sol.y[2],
    label="Infectious I(t)",
)
axes[0].plot(
    state_dates,
    final_sol.y[3],
    label="Removed R(t)",
)

axes[0].axvline(
    BEST_TC_DATE,
    linestyle="--",
    linewidth=1.0,
)

axes[0].set_ylabel("Individuals")
axes[0].set_title(
    "SEIR Compartment Trajectories "
    "(Scale Depends on q)"
)
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(
    state_dates,
    final_sol.y[0],
    label="Susceptible S(t)",
)

axes[1].axvline(
    BEST_TC_DATE,
    linestyle="--",
    linewidth=1.0,
)

axes[1].set_ylabel("Susceptible")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(alpha=0.2)
axes[1].xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "FigS2_final_SEIR_compartments.png",
    dpi=220,
)

plt.close(fig)

print("已保存 FigS2_final_SEIR_compartments.png")


# ============================================================
# 7. Fig 3 — 反事实图
#    保留你喜欢的图，但避免 intervention 因果措辞
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    dates,
    y,
    s=12,
    alpha=0.40,
    label="Observed",
)

ax.plot(
    dates,
    counter_pred,
    "--",
    linewidth=2.0,
    label="Counterfactual: pre-change beta maintained",
)

ax.plot(
    dates,
    piecewise_pred,
    linewidth=2.0,
    label="Fitted free change-point model",
)

ax.axvline(
    BEST_TC_DATE,
    linestyle=":",
    linewidth=1.3,
    label=f"Estimated change point: {BEST_TC_DATE:%b %d}",
)

ax.set_yscale("log")

ax.set_title(
    "Counterfactual With the Pre-Change "
    "Transmission Rate Maintained"
)

ax.set_xlabel("Date")
ax.set_ylabel(
    "Daily reported cases (log scale)"
)

ax.legend()
ax.grid(alpha=0.2)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "FigS3_counterfactual.png",
    dpi=220,
)

plt.close(fig)

print("已保存 FigS3_counterfactual.png")


# ============================================================
# 8. Fig 4 — 生物学敏感性（R + R²）
# ============================================================

bio_plot = bio_df.copy()

bio_plot["Scenario"] = bio_plot.apply(
    lambda r:
        f"L{int(r['Latent_days'])}/"
        f"I{int(r['Infectious_days'])}",
    axis=1,
)

x = np.arange(len(bio_plot))

fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 8),
    sharex=True,
)

axes[0].plot(
    x,
    bio_plot["R_before"],
    "o-",
    label="R before change point",
)

axes[0].plot(
    x,
    bio_plot["R_after"],
    "s-",
    label="R after change point",
)

axes[0].axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="R = 1",
)

axes[0].set_ylabel(
    "Model-implied reproduction index"
)

axes[0].set_title(
    "Sensitivity to Latent and Infectious "
    "Period Assumptions"
)

axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(
    x,
    bio_plot["R2"],
    "o-",
)

axes[1].axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

axes[1].set_ylabel("R²")
axes[1].set_xlabel(
    "Latent / infectious period scenario (days)"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    bio_plot["Scenario"],
    rotation=45,
)

axes[1].grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig4_biological_sensitivity.png",
    dpi=220,
)

plt.close(fig)

print("已保存 Fig4_biological_sensitivity.png")


# ============================================================
# 9. Fig 5 — 最终残差图
# ============================================================

is_weekend = (
    dates.dt.dayofweek.to_numpy() >= 5
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    dates,
    residuals,
    linewidth=1.0,
    label="Residual",
)

ax.scatter(
    dates[is_weekend],
    residuals[is_weekend],
    s=18,
    label="Weekend",
    zorder=3,
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.2,
)

ax.axvline(
    BEST_TC_DATE,
    linestyle=":",
    linewidth=1.2,
    label=f"Estimated change point: {BEST_TC_DATE:%b %d}",
)

ax.set_title(
    "Residual Diagnostics of the "
    "Free Change-Point SEIR Model"
)

ax.set_xlabel("Date")
ax.set_ylabel(
    "Residual (Observed - Predicted)"
)

ax.legend()
ax.grid(alpha=0.2)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig5_residuals.png",
    dpi=220,
)

plt.close(fig)

print("已保存 Fig5_residuals.png")


# ============================================================
# 10. 论文真正需要的附加图 AIC profile
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    profile_df["tc_date"],
    profile_df["AIC"],
    linewidth=1.8,
)

ax.axvline(
    BEST_TC_DATE,
    linestyle="--",
    linewidth=1.3,
    label=f"AIC minimum: {BEST_TC_DATE:%b %d}",
)

ax.axvline(
    observed_peak_date,
    linestyle=":",
    linewidth=1.3,
    label=f"Observed peak: {observed_peak_date:%b %d}",
)

ax.set_title(
    "Data-Driven Change-Point Profile"
)

ax.set_xlabel(
    "Candidate change point"
)

ax.set_ylabel(
    "AIC (lower is better)"
)

ax.legend()
ax.grid(alpha=0.2)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig2_changepoint_profile.png",
    dpi=220,
)

plt.close(fig)

print("已保存 Fig2_changepoint_profile.png")


# ============================================================
# 11. 论文真正需要的附加图 change-point sensitivity
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    cp_sens_df["ChangeDate"],
    cp_sens_df["R_before"],
    marker="o",
    label="R before",
)

ax.plot(
    cp_sens_df["ChangeDate"],
    cp_sens_df["R_after"],
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
    "Change-Point Sensitivity"
)

ax.set_xlabel(
    "Prespecified change point"
)

ax.set_ylabel(
    "Model-implied reproduction index"
)

ax.legend()
ax.grid(alpha=0.2)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d")
)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTDIR / "Fig3_changepoint_sensitivity.png",
    dpi=220,
)

plt.close(fig)

print("已保存 Fig3_changepoint_sensitivity.png")


# ============================================================
# 12. q sensitivity（补充）
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    q_df["q"],
    q_df["R_before"],
    marker="o",
    label="R before",
)

ax.plot(
    q_df["q"],
    q_df["R_after"],
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
    "Observation-Scale Sensitivity"
)

ax.set_xlabel(
    "q (observation scaling factor)"
)

ax.set_ylabel(
    "Model-implied reproduction index"
)

ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUTDIR / "FigS1_q_sensitivity.png",
    dpi=220,
)

plt.close(fig)

print("已保存 FigS1_q_sensitivity.png")


print("\n全部完成。")
print(f"最终图片目录：{OUTDIR.resolve()}")