"""DISASTER-X AI - Flood susceptibility model pipeline v2 (leakage-free).

v2 changes vs v1 (found in self-audit):
- REMOVED leakage features: flood_years and n_detections are both derived from
  the same detection history that defines the label (>=3 flood months). Using
  them makes the task trivially separable (v1 ROC-AUC 0.99 was inflated).
- The model must now predict *recurrence* from static environmental features
  + a *disjoint* history proxy: detections in the FIRST HALF of the record
  (2014-2019) predict recurrence in the SECOND HALF (2020-2024).
  This makes it a genuine temporal generalization task, honestly labelled
  "long-term susceptibility from early-record evidence".
- Label: cell had >= MIN_FLOOD_MONTHS_LATE distinct flood months in 2020-2024.
- Features: static environmental aggregates + early-record statistics only.
- Split: geographic leave-tile-out (train 8 tiles / val 1 / test 2).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, brier_score_loss,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap
import joblib

RNG = 42
SRC = "/home/z/my-project/ml/data/processed/ai4g_sample.parquet"
OUT = "/home/z/my-project/ml/reports"
ART = "/home/z/my-project/ml/models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(ART, exist_ok=True)

SPLIT_YEAR = 2020          # early = 2014-2019, late = 2020-2024
MIN_FLOOD_MONTHS_LATE = 2  # label: >=2 distinct flood months in late period

df = pd.read_parquet(SRC)
for c in ["soil_moisture", "soil_moisture_zscore", "soil_moisture_sca",
          "soil_moisture_zscore_sca", "temp"]:
    df[c] = df[c].where(df[c] > -900)

df["lat_g"] = np.floor(df.lat / 0.05) * 0.05
df["lon_g"] = np.floor(df.lon / 0.05) * 0.05
df["ym"] = df.year * 12 + df.month
early = df[df.year < SPLIT_YEAR]
late = df[df.year >= SPLIT_YEAR]

# --- late-period labels ---
late_agg = late.groupby(["tile_id", "lat_g", "lon_g"]).agg(
    late_flood_months=("ym", "nunique"))
late_agg["label"] = (late_agg.late_flood_months >= MIN_FLOOD_MONTHS_LATE).astype(int)

# --- early-period features (information available BEFORE the label window) ---
early_agg = early.groupby(["tile_id", "lat_g", "lon_g"]).agg(
    early_flood_months=("ym", "nunique"),
    early_n_detections=("ym", "size"),
    early_flood_years=("year", "nunique"),
    dem_metric_1=("dem_metric_1", "mean"),
    dem_metric_2=("dem_metric_2", "mean"),
    dem_metric_1_max=("dem_metric_1", "max"),
    soil_moisture=("soil_moisture", "mean"),
    soil_moisture_std=("soil_moisture", "std"),
    soil_moisture_zscore=("soil_moisture_zscore", "mean"),
    temp=("temp", "mean"),
    edge_fp_rate=("edge_false_positives", "mean"),
    land_cover=("land_cover", lambda s: s.mode().iat[0]),
    lat=("lat", "mean"),
    lon=("lon", "mean"),
).reset_index()

dom = early.groupby(["tile_id", "lat_g", "lon_g", "month"]).size().rename("n").reset_index()
dom = dom.sort_values("n", ascending=False).drop_duplicates(["tile_id", "lat_g", "lon_g"])
early_agg = early_agg.merge(dom[["tile_id", "lat_g", "lon_g", "month"]].rename(
    columns={"month": "dominant_month"}), on=["tile_id", "lat_g", "lon_g"], how="left")

data = early_agg.merge(late_agg.reset_index()[["tile_id", "lat_g", "lon_g", "label"]],
                       on=["tile_id", "lat_g", "lon_g"], how="inner")
print(f"cells with both early and late records: {len(data)}")
print("label balance:", data.label.mean().round(3))
print(data.groupby(["tile_id", "label"]).size().unstack(fill_value=0).to_string())

FEATURES = [
    "early_flood_months", "early_flood_years", "early_n_detections",
    "dem_metric_1", "dem_metric_2", "dem_metric_1_max",
    "soil_moisture", "soil_moisture_std", "soil_moisture_zscore",
    "temp", "land_cover", "dominant_month", "lat", "lon",
]
TARGET = "label"

# ---------------- EDA ----------------
corr = data[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\ncorrelation with label:\n", corr.round(3).to_string())
with open(f"{OUT}/eda_summary_v2.json", "w") as f:
    json.dump({
        "n_cells": int(len(data)),
        "label_balance": data.label.value_counts(normalize=True).round(4).to_dict(),
        "correlation_with_label": corr.round(3).to_dict(),
        "task": f"predict >= {MIN_FLOOD_MONTHS_LATE} flood months in {SPLIT_YEAR}-2024 from 2014-{SPLIT_YEAR-1} evidence",
    }, f, indent=2)

fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
for ax, col in zip(axes.flat, FEATURES):
    for lbl, color, name in [(1, "#4285FF", "recurrent"), (0, "#EAB308", "non-recurrent")]:
        vals = data.loc[data.label == lbl, col].dropna()
        ax.hist(vals, bins=40, alpha=0.55, density=True, color=color, label=name)
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=7)
axes.flat[0].legend(fontsize=7)
fig.suptitle("Feature distributions by late-period flood recurrence (leakage-free task)", fontsize=12)
fig.savefig(f"{OUT}/eda_distributions_v2.png", dpi=100)
plt.close(fig)

# ---------------- Geographic split ----------------
TRAIN_TILES = ["N24E090", "N12E105", "N15E033", "N21E078", "N27E069", "S18E033", "S30E147", "N03E042"]
VAL_TILES = ["N09E006"]
TEST_TILES = ["N33W090", "S09W060"]

train = data[data.tile_id.isin(TRAIN_TILES)]
val = data[data.tile_id.isin(VAL_TILES)]
test = data[data.tile_id.isin(TEST_TILES)]
Xtr, ytr = train[FEATURES], train[TARGET].values
Xva, yva = val[FEATURES], val[TARGET].values
Xte, yte = test[FEATURES], test[TARGET].values
# soil_moisture_std is NaN when a cell has a single early observation
Xtr, Xva, Xte = (X.fillna(-1) for X in (Xtr, Xva, Xte))
print(f"\ntrain={len(train)} (pos {ytr.mean():.2f}) val={len(val)} (pos {yva.mean():.2f}) test={len(test)} (pos {yte.mean():.2f})")

# ---------------- Models ----------------
models = {
    "logreg": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ]),
    "rf": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=2, random_state=RNG),
    "xgb": xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.06, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
        eval_metric="logloss", n_jobs=2, random_state=RNG),
}

results = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    pred = (pte >= 0.5).astype(int)
    results[name] = {
        "val": {"roc_auc": roc_auc_score(yva, pva), "pr_auc": average_precision_score(yva, pva)},
        "test": {
            "roc_auc": roc_auc_score(yte, pte),
            "pr_auc": average_precision_score(yte, pte),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "brier": brier_score_loss(yte, pte),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        },
    }
    r = results[name]["test"]
    print(f"{name}: val AUC={results[name]['val']['roc_auc']:.3f} | test AUC={r['roc_auc']:.3f} "
          f"PR={r['pr_auc']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} Brier={r['brier']:.3f} CM={r['confusion_matrix']}")

# ---------------- Comparison figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
colors = {"logreg": "#94A3B8", "rf": "#25D9F2", "xgb": "#4285FF"}
for name, model in models.items():
    pte = model.predict_proba(Xte)[:, 1]
    RocCurveDisplay.from_predictions(yte, pte, ax=axes[0], name=name, color=colors[name])
    PrecisionRecallDisplay.from_predictions(yte, pte, ax=axes[1], name=name, color=colors[name])
    frac_pos, mean_pred = calibration_curve(yte, pte, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", color=colors[name], label=name)
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_title("ROC - geographic test tiles")
axes[1].set_title("Precision-Recall - test")
axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[2].set_title("Calibration (reliability)")
axes[2].set_xlabel("mean predicted probability")
axes[2].set_ylabel("fraction recurrent")
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(f"{OUT}/model_comparison_v2.png", dpi=110)
plt.close(fig)

# ---------------- Select + SHAP ----------------
best_name = max(results, key=lambda k: results[k]["val"]["roc_auc"])
best = models[best_name]
print(f"\nselected: {best_name}")

sv = explainer_imp = None
if best_name in ("rf", "xgb"):
    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    if np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, Xte, feature_names=FEATURES, show=False, max_display=14)
    plt.title("SHAP summary - test tiles (v2, leakage-free)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary_v2.png", dpi=110)
    plt.close("all")
    mean_abs = np.asarray(np.abs(sv)).mean(axis=0).ravel()
    shap_importance = dict(sorted(zip(FEATURES, mean_abs.round(4)), key=lambda kv: -kv[1]))
    print("SHAP importance:", shap_importance)
else:
    shap_importance = None

# ---------------- Artifacts ----------------
artifact = f"{ART}/flood_susceptibility_v2_{best_name}.joblib"
joblib.dump({"model": best, "features": FEATURES, "target": TARGET,
             "train_tiles": TRAIN_TILES, "val_tiles": VAL_TILES, "test_tiles": TEST_TILES,
             "split_year": SPLIT_YEAR, "min_flood_months_late": MIN_FLOOD_MONTHS_LATE}, artifact)

meta = {
    "model_name": "flood_susceptibility",
    "version": "2.0.0",
    "algorithm": {"logreg": "LogisticRegression", "rf": "RandomForestClassifier",
                  "xgb": "XGBClassifier"}[best_name],
    "selected": best_name,
    "features": FEATURES,
    "task": f"Predict >= {MIN_FLOOD_MONTHS_LATE} distinct flood months in {SPLIT_YEAR}-2024 from 2014-{SPLIT_YEAR-1} evidence and static environmental features",
    "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
    "class_balance": {"train": float(ytr.mean()), "val": float(yva.mean()), "test": float(yte.mean())},
    "results": results,
    "shap_importance": shap_importance,
    "data_provenance": {
        "source": "ai-for-good-lab/ai4g-flood-dataset (HuggingFace, MIT license)",
        "paper": "Mapping global floods with 10 years of satellite radar data, Nature Communications 2025, DOI 10.1038/s41467-025-60973-1",
        "label_definition": f"grid cell (0.05 deg) with >= {MIN_FLOOD_MONTHS_LATE} distinct Sentinel-1 flood detection months in {SPLIT_YEAR}-2024",
        "split": "geographic leave-tile-out AND temporal (features from 2014-2019, label from 2020-2024)",
    },
    "artifact_path": artifact,
}
with open(f"{OUT}/model_report_v2.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nsaved {artifact} and {OUT}/model_report_v2.json")

# also save the modeling dataset for the app / evaluation reuse
data.to_parquet("/home/z/my-project/ml/data/processed/susceptibility_dataset_v2.parquet", index=False)
