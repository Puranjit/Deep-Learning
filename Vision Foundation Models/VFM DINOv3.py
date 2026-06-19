# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:48:30 2026

@author: puran
"""
###
This code extracts Feature embeddings using the DINOv3 model of DINOv3_vitb16 and trains ML models using the LazyPredict Library on those features to train on two target variables (Environment modeling)
### 
# FINAL CODE
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor, REGRESSORS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Daily_averaged_embeddings_Dv3_vitb16
DATA_PATH    = "Daily_averaged_embeddings_Dv3_vitb16.xlsx"
model_label  = DATA_PATH.split("_")[4].split(".")[0]
FEATURE_COLS = [f"dino_cls_{i}" for i in range(772)]

TARGETS      = ["NEE_f", "GPP_f"]

df           = pd.read_excel(DATA_PATH, sheet_name=2)
df["year"]   = pd.to_datetime(df["date"]).dt.year
df["date"]   = pd.to_datetime(df["date"])
years        = sorted(df["year"].unique())

print(f"Years found : {years}")
print(f"Total rows  : {len(df)}")

annual_stats = df.groupby('year')[['GPP_f','NEE_f']].agg(
    ['mean','std','min','max']
)
print(annual_stats)

import seaborn as sns

sns.boxplot(data=df, x='year', y='GPP_f')
plt.show()

sns.boxplot(data=df, x='year', y='NEE_f')
plt.show()

# ─────────────────────────────────────────────
# LOYO-CV — track ALL models + collect predictions
# from the globally best model (highest mean R²)
# ─────────────────────────────────────────────
def run_loyo(df, target: str):
    """
    Pass 1 — run LazyPredict on every fold, store all model scores.
    Pass 2 — identify globally best model (highest mean R² across folds),
             refit it on every fold and collect actual vs predicted.

    Returns
    -------
    all_results  : DataFrame  — every model × every fold
    fold_summary : DataFrame  — best model per fold (R², RMSE, MAE)
    top5         : DataFrame  — top-5 models by mean R²
    best_name    : str        — globally best model name
    pred_df      : DataFrame  — date / year / actual / predicted
    best_metrics : dict       — per fold {R2, RMSE, MAE} for best model
    """

    # ── PASS 1 : LazyPredict on all folds ────────────────────────────
    all_fold_dfs  = []
    fold_bests    = []

    print("=" * 70)
    print(f"PASS 1 — LazyPredict LOYO-CV   TARGET: {target}")
    print("=" * 70)

    for test_year in years:
        train_years = [y for y in years if y != test_year]
        train_df    = df[df["year"].isin(train_years)].copy()
        test_df     = df[df["year"] == test_year].copy()

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[target].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df[target].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        reg          = LazyRegressor(verbose=0, ignore_warnings=True,
                                     predictions=False)
        models_df, _ = reg.fit(X_train, X_test, y_train, y_test)

        # tag fold
        models_df            = models_df.copy().reset_index()
        models_df.rename(columns={"index": "Model"}, inplace=True)
        models_df["Test_Year"]  = test_year
        models_df["Target"]     = target
        all_fold_dfs.append(models_df)

        # best this fold
        best_row = models_df.sort_values("R-Squared", ascending=False).iloc[0]
        fold_bests.append({
            "Test_Year" : test_year,
            "Best_Model": best_row["Model"],
            "R2"        : round(best_row["R-Squared"], 4),
            "RMSE"      : round(best_row["RMSE"],      4),
        })

        print(f"  Fold {test_year} | "
              f"Best: {best_row['Model']:<40} "
              f"R²={best_row['R-Squared']:.4f}  "
              f"RMSE={best_row['RMSE']:.4f}")

    all_results  = pd.concat(all_fold_dfs, ignore_index=True)
    fold_summary = pd.DataFrame(fold_bests)

    # ── Global ranking — mean R² across all folds ────────────────────
    global_rank = (
        all_results.groupby("Model")[["R-Squared", "RMSE"]]
        .mean()
        .rename(columns={"R-Squared": "Mean_R2", "RMSE": "Mean_RMSE"})
        .sort_values("Mean_R2", ascending=False)
        .reset_index()
    )
    top5      = global_rank.head(5)
    best_name = global_rank.iloc[0]["Model"]

    print(f"\n  🏆 Globally best model (mean R²): {best_name}")
    print(f"\n  Top-5 models by mean R²:")
    print(top5.to_string(index=False))

    # ── PASS 2 : refit best model on every fold → predictions ────────
    print(f"\n{'=' * 70}")
    print(f"PASS 2 — Refit [{best_name}] on every fold")
    print("=" * 70)

    best_cls     = next(cls for name, cls in REGRESSORS if name == best_name)
    pred_records = []
    best_metrics = {}

    for test_year in years:
        train_years = [y for y in years if y != test_year]
        train_df    = df[df["year"].isin(train_years)].copy()
        test_df     = df[df["year"] == test_year].copy()

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[target].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df[target].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        mdl = best_cls()
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        best_metrics[test_year] = {
            "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)
        }

        print(f"  Fold {test_year} | "
              f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        for date, actual, predicted in zip(test_df["date"], y_test, y_pred):
            pred_records.append({
                "date": date, "year": test_year,
                "actual": actual, "predicted": predicted,
            })

    pred_df = (pd.DataFrame(pred_records)
               .sort_values("date")
               .reset_index(drop=True))

    mean_r2   = np.mean([v["R2"]   for v in best_metrics.values()])
    mean_rmse = np.mean([v["RMSE"] for v in best_metrics.values()])
    mean_mae  = np.mean([v["MAE"]  for v in best_metrics.values()])
    print(f"\n  Overall — Mean R²={mean_r2:.4f}  "
          f"Mean RMSE={mean_rmse:.4f}  Mean MAE={mean_mae:.4f}")

    return all_results, fold_summary, top5, best_name, pred_df, best_metrics

# ─────────────────────────────────────────────
# RUN FOR BOTH TARGETS
# ─────────────────────────────────────────────
results_store = {}

for target in TARGETS:
    out = run_loyo(df, target)
    results_store[target] = {
        "all_results"  : out[0],
        "fold_summary" : out[1],
        "top5"         : out[2],
        "best_name"    : out[3],
        "pred_df"      : out[4],
        "best_metrics" : out[5],
    }

# ─────────────────────────────────────────────
# PLOT — best model line plots
# ─────────────────────────────────────────────
COLORS = {"actual": "#2c7bb6", "predicted": "#d7191c"}

def plot_loyo_lineplots(results_store, target):
    store     = results_store[target]
    pred_df   = store["pred_df"]
    metrics   = store["best_metrics"]
    best_name = store["best_name"]
    n_years   = len(years)

    mean_r2   = np.mean([v["R2"]   for v in metrics.values()])
    mean_rmse = np.mean([v["RMSE"] for v in metrics.values()])
    mean_mae  = np.mean([v["MAE"]  for v in metrics.values()])

    # ── Precipitation column from feature embeddings ──────────────
    # PRECIP_COL = "dino_cls_385"   # ← adjust index if needed
    PRECIP_COL = "dino_cls_769"   # ← adjust index if needed
    # PRECIP_COL = "dino_cls_1025"   # ← adjust index if needed
    # PRECIP_COL = "dino_cls_1281"   # ← adjust index if needed
    
    fig, axes = plt.subplots(
        nrows=n_years, ncols=1,
        figsize=(16, 3.5 * n_years),
        sharex=False
    )
    if n_years == 1:
        axes = [axes]

    # ── Main title — overall metrics in bold ──────────────────────
    fig.suptitle(
        f"LOYO-CV  |  {target}\n"
        f"Best Model: {best_name}  |  "
        f"Mean R² = {mean_r2:.4f}   "
        f"Mean RMSE = {mean_rmse:.4f}   "
        f"Mean MAE = {mean_mae:.4f}",
        fontsize=12,
        fontweight="bold",
        y=1.01
    )

    for ax, test_year in zip(axes, years):
        yr_df = pred_df[pred_df["year"] == test_year].sort_values("date")
        m     = metrics[test_year]

        # ── Pull precipitation for this test year from df ─────────
        yr_raw = df[df["year"] == test_year].sort_values("date")

        # ── Left axis — actual vs predicted ──────────────────────
        ax.plot(yr_df["date"], yr_df["actual"],
                color=COLORS["actual"], lw=1.5,
                label="Actual", alpha=0.9)
        ax.plot(yr_df["date"], yr_df["predicted"],
                color=COLORS["predicted"], lw=1.5,
                label="Predicted", alpha=0.85, linestyle="--")
        ax.fill_between(yr_df["date"],
                        yr_df["actual"], yr_df["predicted"],
                        alpha=0.12, color="grey")

        ax.set_ylabel(target, fontsize=16, fontweight= "bold")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        # ── Per-fold title — metrics in bold ─────────────────────
        ax.set_title(
            f"{test_year} (test fold)   "
            f"R² = {m['R2']:.3f}   "
            f"RMSE = {m['RMSE']:.3f}   "
            f"MAE = {m['MAE']:.3f}   "
            f"Model: {best_name}",
            fontsize=12,
            fontweight="bold",
            loc="left"
        )

        # ── Right axis — precipitation ────────────────────────────
        ax2 = ax.twinx()

        if PRECIP_COL in yr_raw.columns:
            ax2.bar(yr_raw["date"], yr_raw[PRECIP_COL],
                    color="lightcoral", alpha=0.6,
                    width=5.0, label="Precipitation")
            ax2.set_ylim(0, 1.3)
            ax2.set_ylabel("Precipitation (in)", fontsize=16, fontweight = "bold", color="black")
            ax2.tick_params(axis="y", labelsize=10, labelcolor="black")
            ax2.spines["right"].set_color("black")
        else:
            print(f"  WARNING: '{PRECIP_COL}' not found for {test_year}")

        # ── Combined legend (both axes) ───────────────────────────
        lines_left,  labels_left  = ax.get_legend_handles_labels()
        lines_right, labels_right = ax2.get_legend_handles_labels()
        ax.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            fontsize=8, loc="center right", framealpha=0.6
        )

    plt.tight_layout()
    out_path = f"loyo_{target}_{best_name}_{model_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()

for target in TARGETS:
    plot_loyo_lineplots(results_store, target)

# ─────────────────────────────────────────────
# SAVE TO EXCEL
# ─────────────────────────────────────────────
# OUTPUT_FILE = f"cv_results_{model_label}_Weather.xlsx"
OUTPUT_FILE = f"cv_results_" + model_label + ".xlsx"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for target in TARGETS:
        store     = results_store[target]
        best_name = store["best_name"]

        # All models — all folds
        store["all_results"].to_excel(
            writer, sheet_name=f"{target}_all_models", index=False)

        # Global ranking (mean R² across folds)
        (store["all_results"]
         .groupby("Model")[["R-Squared", "RMSE"]]
         .mean()
         .rename(columns={"R-Squared": "Mean_R2", "RMSE": "Mean_RMSE"})
         .sort_values("Mean_R2", ascending=False)
         .reset_index()
         .to_excel(writer, sheet_name=f"{target}_global_ranking", index=False))

        # Top-5
        store["top5"].to_excel(
            writer, sheet_name=f"{target}_top5", index=False)

        # Best model per fold (from LazyPredict sweep)
        store["fold_summary"].to_excel(
            writer, sheet_name=f"{target}_best_per_fold", index=False)

        # Best global model predictions + metrics
        metrics_df = pd.DataFrame(store["best_metrics"]).T.reset_index()
        metrics_df.columns = ["Test_Year", "R2", "RMSE", "MAE"]
        metrics_df["Best_Model"] = best_name
        metrics_df.to_excel(
            writer, sheet_name=f"{target}_best_model_metrics", index=False)

        store["pred_df"].to_excel(
            writer, sheet_name=f"{target}_predictions", index=False)

print(f"\n✓ Results saved → {OUTPUT_FILE}")
