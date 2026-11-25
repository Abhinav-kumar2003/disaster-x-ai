"""DISASTER-X AI - Flood susceptibility model pipeline (Task B).

Dataset: grid cells from AI4G Sentinel-1 flood detections (11 tiles worldwide).
Label: recurrent flood cell (>= 3 distinct flood months over 2014-2024) vs not.
Split: geographic (leave-tile-out) - train on 8 tiles, validate on 1, test on 2,
so no spatial leakage between train and eval.
Models: Logistic Regression baseline, Random Forest, XGBoost.
Evaluation: precision/recall/F1/ROC-AUC/PR-AUC, confusion matrix,
calibration (Brier + reliability curve), false-negative analysis, SHAP.
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
                             confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import shap
import joblib

RNG = 42
DATA = "/home/z/my-project/ml/data/processed/grid_dataset.parquet"
OUT = "/home/z/my-project/ml/reports"
ART = "/home/z/my-project/ml/models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(ART, exist_ok=True)

df = pd.read_parquet(DATA)

FEATURES = [
    "dem_metric_1", "dem_metric_2", "dem_metric_1_max",
    "soil_moisture", "soil_moisture_std", "soil_moisture_zscore",
    "temp", "land_cover", "dominant_month", "lat", "lon",
    "flood_years", "n_detections",
]
# NOTE: `flood_months` is the label source and must NOT be a feature.
TARGET = "label"

# ---------------- EDA ----------------
eda = {
    "n_cells": int(len(df)),
    "label_balance": df.label.value_counts(normalize=True).round(4).to_dict(),
    "features_describe": df[FEATURES].describe().round(3).to_dict(),
    "missing": df[FEATURES].isna().sum().astype(int).to_dict(),
    "label_by_landcover": df.groupby("land_cover").label.mean().round(3).to_dict(),
    "label_by_tile": df.groupby("tile_id").label.mean().round(3).to_dict(),
}
# correlations with label
corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
eda["correlation_with_label"] = corr.round(3).to_dict()
with open(f"{OUT}/eda_summary.json", "w") as f:
    json.dump(eda, f, indent=2, default=str)
print("EDA: correlation with label:\n", corr.round(3).to_string())

# EDA figure: feature distributions by class
fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
for ax, col in zip(axes.flat, FEATURES):
    for lbl, color, name in [(1, "#4285FF", "recurrent"), (0, "#EAB308", "non-recurrent")]:
        vals = df.loc[df.label == lbl, col].dropna()
        ax.hist(vals, bins=40, alpha=0.55, density=True, color=color, label=name)
    ax.set_title(col, fontsize=10)
    ax.tick_params(labelsize=8)
axes.flat[0].legend(fontsize=8)
fig.suptitle("Feature distributions by flood-recurrence class (AI4G grid cells)", fontsize=12)
fig.savefig(f"{OUT}/eda_distributions.png", dpi=110)
plt.close(fig)

# ---------------- Geographic split ----------------
TRAIN_TILES = ["N24E090", "N12E105", "N15E033", "N21E078", "N27E069", "S18E033", "S30E147", "N03E042"]
VAL_TILES = ["N09E006"]
TEST_TILES = ["N33W090", "S09W060"]

train = df[df.tile_id.isin(TRAIN_TILES)]
val = df[df.tile_id.isin(VAL_TILES)]
test = df[df.tile_id.isin(TEST_TILES)]
print(f"\ntrain={len(train)} val={len(val)} test={len(test)}")
print("train balance:", train.label.mean().round(3), "| val:", val.label.mean().round(3), "| test:", test.label.mean().round(3))

Xtr, ytr = train[FEATURES].fillna(-1), train[TARGET].values
Xva, yva = val[FEATURES].fillna(-1), val[TARGET].values
Xte, yte = test[FEATURES].fillna(-1), test[TARGET].values

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
        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
        eval_metric="logloss", n_jobs=2, random_state=RNG),
}

results = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    res = {
        "val": {
            "roc_auc": roc_auc_score(yva, pva),
            "pr_auc": average_precision_score(yva, pva),
        },
        "test": {},
    }
    for split, X, y, p in [("test", Xte, yte, pte)]:
        pred = (p >= 0.5).astype(int)
        res[split] = {
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "brier": brier_score_loss(y, p),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
    results[name] = res
    print(f"\n{name}: val ROC-AUC={res['val']['roc_auc']:.3f} PR-AUC={res['val']['pr_auc']:.3f}")
    print(f"  test ROC-AUC={res['test']['roc_auc']:.3f} PR-AUC={res['test']['pr_auc']:.3f} "
          f"P={res['test']['precision']:.3f} R={res['test']['recall']:.3f} F1={res['test']['f1']:.3f} "
          f"Brier={res['test']['brier']:.3f}")
    print(f"  CM={res['test']['confusion_matrix']}")

# ---------------- Model comparison figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
colors = {"logreg": "#94A3B8", "rf": "#25D9F2", "xgb": "#4285FF"}
for name, model in models.items():
    pte = model.predict_proba(Xte)[:, 1]
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    RocCurveDisplay.from_predictions(yte, pte, ax=axes[0], name=name, color=colors[name])
    PrecisionRecallDisplay.from_predictions(yte, pte, ax=axes[1], name=name, color=colors[name])
    frac_pos, mean_pred = calibration_curve(yte, pte, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", color=colors[name], label=name)
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_title("ROC - test (geographic holdout)")
axes[1].set_title("Precision-Recall - test")
axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[2].set_title("Calibration (reliability)")
axes[2].set_xlabel("mean predicted probability")
axes[2].set_ylabel("fraction recurrent")
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(f"{OUT}/model_comparison.png", dpi=110)
plt.close(fig)

# ---------------- Select best by validation ROC-AUC ----------------
best_name = max(results, key=lambda k: results[k]["val"]["roc_auc"])
best = models[best_name]
print(f"\nselected model: {best_name}")

# ---------------- SHAP explanations ----------------
if best_name in ("rf", "xgb"):
    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    if np.ndim(sv) == 3:  # (n, features, classes) -> positive class
        sv = sv[:, :, 1]
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, Xte, feature_names=FEATURES, show=False, max_display=12)
    plt.title("SHAP summary - test tiles")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary.png", dpi=110)
    plt.close(fig)
    mean_abs = np.asarray(np.abs(sv)).mean(axis=0).ravel()
    shap_importance = dict(sorted(zip(FEATURES, mean_abs.round(4)), key=lambda kv: -kv[1]))
    print("SHAP importance:\n", shap_importance)
else:
    sv = None
    shap_importance = None

# ---------------- Save artifacts + metadata ----------------
artifact = f"{ART}/flood_susceptibility_{best_name}.joblib"
joblib.dump({"model": best, "features": FEATURES, "target": TARGET,
             "train_tiles": TRAIN_TILES, "val_tiles": VAL_TILES, "test_tiles": TEST_TILES}, artifact)

meta = {
    "model_name": "flood_susceptibility",
    "version": "1.0.0",
    "algorithm": {"logreg": "LogisticRegression", "rf": "RandomForestClassifier",
                  "xgb": "XGBClassifier"}[best_name],
    "selected": best_name,
    "features": FEATURES,
    "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
    "class_balance_train": float(ytr.mean()),
    "results": results,
    "shap_importance": shap_importance,
    "data_provenance": {
        "source": "ai-for-good-lab/ai4g-flood-dataset (HuggingFace, MIT license)",
        "paper": "Mapping global floods with 10 years of satellite radar data, Nature Communications 2025, DOI 10.1038/s41467-025-60973-1",
        "label_definition": "grid cell (0.05deg) with >= 3 distinct Sentinel-1 flood detection months 2014-2024",
        "split": "geographic leave-tile-out",
    },
    "artifact_path": artifact,
}
with open(f"{OUT}/model_report.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nsaved {artifact} and {OUT}/model_report.json")
