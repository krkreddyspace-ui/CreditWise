"""
CreditWise — Exploratory Data Analysis
========================================
Generates and saves all EDA plots and findings to reports/figures/.
Run this script to produce all EDA outputs before building notebooks.

Usage:
    python -m src.eda
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import FIGURES_DIR, RAW_DATA_FILE, TARGET_COLUMN
from src.data_loader import load_raw_data, get_dataset_summary

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "figure.dpi": 140,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE_BINARY = ["#38bdf8", "#f43f5e"]   # 0 = blue, 1 = red


def savefig(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / f"eda_{name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    logger.info("Saved: %s", p.name)
    return p


# ===========================================================================
# 1. BASIC DATASET INFO
# ===========================================================================

def plot_dataset_overview(df: pd.DataFrame, summary: dict) -> None:
    """Print and save a quick overview table."""
    logger.info("\n=== DATASET OVERVIEW ===")
    logger.info("Shape          : %d rows × %d columns", *df.shape)
    logger.info("Duplicate rows : %d", summary["duplicate_rows"])
    logger.info("Target dist    : %s", summary["target_distribution"])
    logger.info("Imbalance ratio: %.2f : 1", summary["target_imbalance_ratio"])
    logger.info("\n%s", df.dtypes.to_string())
    logger.info("\n%s", df.describe().round(2).to_string())


# ===========================================================================
# 2. TARGET CLASS DISTRIBUTION
# ===========================================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    pcts   = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    labels = ["No Default (0)", "Default (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=PALETTE_BINARY, width=0.5, edgecolor="#0f172a")
    axes[0].set_title("Target Class Distribution — Count")
    axes[0].set_ylabel("Number of Borrowers")
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                     f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].grid(axis="y", alpha=0.4)

    # Pie chart
    wedge_props = {"edgecolor": "#0f172a", "linewidth": 2}
    axes[1].pie(
        counts.values, labels=labels, colors=PALETTE_BINARY,
        autopct="%1.1f%%", startangle=90, wedgeprops=wedge_props,
        textprops={"color": "#e2e8f0"},
    )
    axes[1].set_title("Target Class Distribution — Proportion")
    plt.suptitle("SeriousDlqin2yrs — Class Imbalance (13.96 : 1)", fontsize=14, y=1.02)
    plt.tight_layout()
    savefig("01_target_distribution")


# ===========================================================================
# 3. MISSING VALUES
# ===========================================================================

def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#f43f5e" if v > 10 else "#f59e0b" for v in missing_pct.values]
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="#0f172a")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("Missing Value Percentage (%)")
    ax.set_title("Missing Values by Feature")
    ax.set_xlim(0, max(missing_pct.values) * 1.2)
    ax.axvline(5, color="#64748b", linestyle="--", lw=1, label="5% threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    savefig("02_missing_values")


# ===========================================================================
# 4. DESCRIPTIVE STATISTICS — NUMERICAL DISTRIBUTIONS (HISTOGRAMS)
# ===========================================================================

def plot_feature_histograms(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    n_cols = 3
    n_rows = int(np.ceil(len(feature_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()

        # Use log scale if data spans many orders of magnitude
        use_log = data.max() > 1000 or data.skew() > 5
        if use_log:
            data_plot = np.log1p(data.clip(lower=0))
            label = f"log1p({col})"
        else:
            data_plot = data
            label = col

        ax.hist(data_plot, bins=60, color="#38bdf8", edgecolor="#0f172a", alpha=0.85)
        ax.set_title(col, fontsize=10, pad=4)
        if use_log:
            ax.set_xlabel("(log1p scale)", fontsize=8)
        mean_val = data_plot.mean()
        ax.axvline(mean_val, color="#f59e0b", linestyle="--", lw=1.5, label=f"mean={mean_val:.2f}")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.4)

    # Hide unused axes
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions (histograms)", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig("03_feature_histograms")


# ===========================================================================
# 5. OUTLIER ANALYSIS — BOX PLOTS
# ===========================================================================

def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    """Box plots on log1p scale to make extreme outliers visible."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()
        data_log = np.log1p(data.clip(lower=0))

        bp = ax.boxplot(
            data_log, vert=True, patch_artist=True,
            boxprops={"facecolor": "#1d4ed8", "edgecolor": "#38bdf8"},
            medianprops={"color": "#f59e0b", "linewidth": 2},
            whiskerprops={"color": "#94a3b8"},
            capprops={"color": "#94a3b8"},
            flierprops={"marker": ".", "color": "#f43f5e", "markersize": 2, "alpha": 0.3},
        )
        ax.set_title(col, fontsize=9, pad=3)
        ax.set_xlabel("log1p scale", fontsize=7)

        # Annotate raw stats
        n_outliers = (np.abs(stats.zscore(data_log)) > 3).sum()
        ax.text(1.1, ax.get_ylim()[1], f"outliers≈{n_outliers:,}", fontsize=7,
                color="#f43f5e", ha="center", va="top")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Outlier Analysis — Box Plots (log1p scale)", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig("04_outlier_boxplots")


# ===========================================================================
# 6. CORRELATION HEATMAP
# ===========================================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#0f172a",
        annot_kws={"size": 8},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap (Pearson)", fontsize=14, pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    savefig("05_correlation_heatmap")


# ===========================================================================
# 7. DEFAULT RATE BY FEATURE QUARTILE
# ===========================================================================

def plot_default_rates_by_feature(df: pd.DataFrame) -> None:
    """
    For each numerical feature, bin into quartiles and show the default
    rate per bin. This reveals which feature ranges are associated with
    higher/lower observed default rates.
    """
    feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN]]
    n_cols = 2
    n_rows = int(np.ceil(len(feature_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        temp = df[[col, TARGET_COLUMN]].dropna()

        try:
            temp["bin"] = pd.qcut(temp[col], q=4, duplicates="drop")
        except ValueError:
            # Too few unique values — use value_counts binning
            temp["bin"] = temp[col].clip(upper=10)

        default_rate = temp.groupby("bin", observed=True)[TARGET_COLUMN].mean() * 100
        counts = temp.groupby("bin", observed=True)[TARGET_COLUMN].count()

        x = range(len(default_rate))
        bars = ax.bar(x, default_rate.values, color="#38bdf8", edgecolor="#0f172a", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(b) for b in default_rate.index],
            fontsize=7, rotation=30, ha="right"
        )
        ax.set_ylabel("Default Rate (%)", fontsize=8)
        ax.set_title(f"Default Rate by {col}", fontsize=9)
        ax.axhline(df[TARGET_COLUMN].mean() * 100, color="#f59e0b",
                   linestyle="--", lw=1.5, label=f"Overall: {df[TARGET_COLUMN].mean()*100:.1f}%")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.4)

        for bar, c in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"n={c:,}", ha="center", va="bottom", fontsize=6, color="#94a3b8")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Observed Default Rate by Feature Quartile", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("06_default_rates_by_feature")


# ===========================================================================
# 8. FEATURE DISTRIBUTIONS BY TARGET (KDE / violin)
# ===========================================================================

def plot_distributions_by_target(df: pd.DataFrame) -> None:
    """KDE plots showing feature distributions for defaulters vs non-defaulters."""
    feature_cols = ["RevolvingUtilizationOfUnsecuredLines", "age",
                    "DebtRatio", "MonthlyIncome",
                    "NumberOfTime30-59DaysPastDueNotWorse",
                    "NumberOfTimes90DaysLate"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    target_labels = {0: "No Default (0)", 1: "Default (1)"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for t_val, color in zip([0, 1], PALETTE_BINARY):
            subset = df.loc[df[TARGET_COLUMN] == t_val, col].dropna()
            subset_log = np.log1p(subset.clip(lower=0))
            ax.hist(subset_log, bins=50, alpha=0.55, color=color,
                    label=target_labels[t_val], density=True, edgecolor="none")
            # KDE overlay
            from scipy.stats import gaussian_kde
            if len(subset_log.unique()) > 5:
                try:
                    kde = gaussian_kde(subset_log)
                    xs = np.linspace(subset_log.min(), subset_log.max(), 200)
                    ax.plot(xs, kde(xs), color=color, lw=2)
                except Exception:
                    pass

        ax.set_title(f"{col}\n(log1p scale)", fontsize=9)
        ax.set_xlabel("log1p value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Feature Distributions: Defaulters vs Non-Defaulters", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("07_distributions_by_target")


# ===========================================================================
# 9. DELINQUENCY FEATURE ANALYSIS
# ===========================================================================

def plot_delinquency_analysis(df: pd.DataFrame) -> None:
    """Analyse the extreme values (96, 98) in past-due columns."""
    delay_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, delay_cols):
        vc = df[col].value_counts().sort_index()
        # Focus on 0–15 range + flag suspicious high values
        normal = vc[vc.index <= 15]
        suspicious = df[col].isin([96, 98]).sum()

        ax.bar(normal.index, normal.values, color="#38bdf8", edgecolor="#0f172a")
        ax.set_title(f"{col}\n(values > 15: n={suspicious:,} suspicious)", fontsize=9)
        ax.set_xlabel("Count (capped at 15)")
        ax.set_ylabel("Number of Borrowers")
        ax.grid(axis="y", alpha=0.4)

        # Annotate zero vs nonzero
        zero = (df[col] == 0).sum()
        nonzero = (df[col] > 0).sum()
        ax.text(0.97, 0.95, f"Zero: {zero:,}\nNon-zero: {nonzero:,}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox={"facecolor": "#0f172a", "alpha": 0.7})

    plt.suptitle("Delinquency Feature Value Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("08_delinquency_analysis")


# ===========================================================================
# 10. AGE DISTRIBUTION
# ===========================================================================

def plot_age_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Age histogram
    axes[0].hist(df["age"].dropna(), bins=40, color="#818cf8", edgecolor="#0f172a", alpha=0.85)
    axes[0].axvline(df["age"].median(), color="#f59e0b", linestyle="--",
                    lw=2, label=f"Median: {df['age'].median():.0f}")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.4)

    # Default rate by age bracket
    bins = [0, 30, 40, 50, 60, 70, 120]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df_age = df[["age", TARGET_COLUMN]].dropna().copy()
    df_age["age_group"] = pd.cut(df_age["age"], bins=bins, labels=labels, right=False)
    dr = df_age.groupby("age_group", observed=True)[TARGET_COLUMN].mean() * 100

    axes[1].bar(dr.index, dr.values, color="#818cf8", edgecolor="#0f172a", alpha=0.85)
    axes[1].axhline(df[TARGET_COLUMN].mean() * 100, color="#f59e0b",
                    linestyle="--", lw=2, label=f"Overall: {df[TARGET_COLUMN].mean()*100:.1f}%")
    axes[1].set_title("Default Rate by Age Bracket")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.4)

    plt.suptitle("Age Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("09_age_analysis")


# ===========================================================================
# 11. INCOME ANALYSIS
# ===========================================================================

def plot_income_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    income = df["MonthlyIncome"].dropna()
    income_log = np.log1p(income)

    axes[0].hist(income_log, bins=60, color="#34d399", edgecolor="#0f172a", alpha=0.85)
    axes[0].axvline(income_log.median(), color="#f59e0b", linestyle="--",
                    lw=2, label=f"Median: ${np.expm1(income_log.median()):,.0f}/mo")
    axes[0].set_title("Monthly Income Distribution (log1p)")
    axes[0].set_xlabel("log1p(Monthly Income)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.4)

    # Income vs default
    for t_val, color in zip([0, 1], PALETTE_BINARY):
        subset = df.loc[df[TARGET_COLUMN] == t_val, "MonthlyIncome"].dropna()
        axes[1].hist(np.log1p(subset), bins=60, alpha=0.55, color=color,
                     label=f"Class {t_val}", density=True, edgecolor="none")

    axes[1].set_title("Monthly Income: Defaulters vs Non-Defaulters")
    axes[1].set_xlabel("log1p(Monthly Income)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.4)

    plt.suptitle("Monthly Income Analysis (19.8% missing — to be median-imputed)", fontsize=12, y=1.01)
    plt.tight_layout()
    savefig("10_income_analysis")


# ===========================================================================
# 12. DEBT RATIO ANALYSIS
# ===========================================================================

def plot_debt_ratio_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    debt = df["DebtRatio"]
    pct_over_1 = (debt > 1).mean() * 100
    pct_extreme = (debt > 100).mean() * 100

    debt_log = np.log1p(debt.clip(lower=0))
    axes[0].hist(debt_log, bins=80, color="#fb923c", edgecolor="#0f172a", alpha=0.85)
    axes[0].set_title(f"Debt Ratio Distribution (log1p)\n"
                      f"{pct_over_1:.1f}% > 1.0 | {pct_extreme:.1f}% > 100 (extreme)")
    axes[0].set_xlabel("log1p(Debt Ratio)")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.4)

    # Debt ratio vs default (capped)
    for t_val, color in zip([0, 1], PALETTE_BINARY):
        subset = df.loc[df[TARGET_COLUMN] == t_val, "DebtRatio"].clip(upper=5)
        axes[1].hist(subset, bins=60, alpha=0.55, color=color,
                     label=f"Class {t_val}", density=True, edgecolor="none")

    axes[1].set_title("Debt Ratio (capped at 5): Defaulters vs Non-Defaulters")
    axes[1].set_xlabel("Debt Ratio")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.4)

    plt.suptitle("Debt Ratio Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("11_debt_ratio_analysis")


# ===========================================================================
# 13. REVOLVING UTILISATION
# ===========================================================================

def plot_utilisation_analysis(df: pd.DataFrame) -> None:
    util = df["RevolvingUtilizationOfUnsecuredLines"]
    pct_over_1 = (util > 1).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Full distribution (log)
    util_log = np.log1p(util.clip(lower=0))
    axes[0].hist(util_log, bins=80, color="#a78bfa", edgecolor="#0f172a", alpha=0.85)
    axes[0].set_title(f"Revolving Utilization (log1p)\n{pct_over_1:.1f}% > 1.0 (over-limit)")
    axes[0].set_xlabel("log1p(Utilization)")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.4)

    # Capped at 2 for clearer view
    for t_val, color in zip([0, 1], PALETTE_BINARY):
        subset = df.loc[df[TARGET_COLUMN] == t_val, "RevolvingUtilizationOfUnsecuredLines"].clip(upper=2)
        axes[1].hist(subset, bins=60, alpha=0.55, color=color,
                     label=f"Class {t_val}", density=True, edgecolor="none")

    axes[1].set_title("Revolving Utilization (capped at 2): Default vs No Default")
    axes[1].set_xlabel("Revolving Utilization")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.4)

    plt.suptitle("Revolving Utilization Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("12_utilisation_analysis")


# ===========================================================================
# 14. EDA SUMMARY TABLE
# ===========================================================================

def generate_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "Feature": col,
            "Type": str(s.dtype),
            "Missing": s.isna().sum(),
            "Missing%": round(s.isna().mean() * 100, 2),
            "Mean": round(s.mean(), 3) if s.dtype != "object" else "—",
            "Median": round(s.median(), 3) if s.dtype != "object" else "—",
            "Std": round(s.std(), 3) if s.dtype != "object" else "—",
            "Min": round(s.min(), 3) if s.dtype != "object" else "—",
            "Max": round(s.max(), 3) if s.dtype != "object" else "—",
            "Skewness": round(s.skew(), 3) if s.dtype != "object" else "—",
            "Unique": s.nunique(),
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    out_path = Path("reports/results/eda_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info("EDA summary table saved: %s", out_path)
    return summary_df


# ===========================================================================
# MAIN
# ===========================================================================

def run_eda() -> None:
    logger.info("=== CreditWise EDA ===")
    df = load_raw_data()
    summary = get_dataset_summary(df)

    plot_dataset_overview(df, summary)
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_feature_histograms(df)
    plot_outlier_boxplots(df)
    plot_correlation_heatmap(df)
    plot_default_rates_by_feature(df)
    plot_distributions_by_target(df)
    plot_delinquency_analysis(df)
    plot_age_analysis(df)
    plot_income_analysis(df)
    plot_debt_ratio_analysis(df)
    plot_utilisation_analysis(df)
    eda_summary = generate_eda_summary(df)

    logger.info("\n=== KEY EDA FINDINGS ===")
    logger.info("1. Severe class imbalance: 93.3%% vs 6.7%% (ratio 13.96:1)")
    logger.info("2. MonthlyIncome: 19.82%% missing → median imputation required")
    logger.info("3. NumberOfDependents: 2.62%% missing → median imputation")
    logger.info("4. RevolvingUtilization: %.1f%% values > 1.0 (extreme outliers)",
                (df["RevolvingUtilizationOfUnsecuredLines"] > 1).mean() * 100)
    logger.info("5. DebtRatio: %.1f%% values > 100 (data quality issue, not real)",
                (df["DebtRatio"] > 100).mean() * 100)
    logger.info("6. Delinquency columns contain suspicious values 96/98 (missing codes)")
    logger.info("7. 609 duplicate rows detected")
    logger.info("8. age=0 present: %d rows", (df["age"] == 0).sum())
    logger.info("All EDA figures saved to reports/figures/")
    return df, summary


if __name__ == "__main__":
    run_eda()
