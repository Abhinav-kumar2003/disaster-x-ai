# DISASTER-X AI — Verified ML Core + Inference Layer (Phases 1–3)

**Geospatial Disaster Intelligence Platform — "Intelligence Before Impact."**

This package contains the **verified, reproducible AI/ML core** of DISASTER-X AI, the **Phase 2 inference layer** (FastAPI service, calibrated alert engine, PostGIS schema, dashboards), and the **Phase 3 completion** (landslide module on NASA/Landslide4Sense data, extended API, React frontend source, Docker orchestration, Alembic migrations, and the full documentation set). Every number in this README was produced by the scripts included here and can be re-verified by running them.

---

## 1. What was built and verified in this phase

### 1.1 Flood susceptibility model (tabular ML) — OPERATIONAL
- **Task (leakage-free):** predict whether a 0.05° grid cell will experience ≥ 2 distinct Sentinel-1 flood-detection months in **2020–2024**, using only evidence from **2014–2019** plus static environmental features (terrain metrics, soil moisture, temperature, land cover, location).
- **Data:** [ai-for-good-lab/ai4g-flood-dataset](https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset) (MIT license) — Sentinel-1 SAR flood detections 2014–2024, from *Mapping global floods with 10 years of satellite radar data*, Nature Communications 2025, DOI 10.1038/s41467-025-60973-1. 11 globally distributed tiles (~6M sampled detection rows → 20,386 grid cells).
- **Split:** geographic leave-tile-out (train 8 tiles: Bangladesh, Cambodia, Sudan, India, Pakistan, Mozambique, Australia, Ethiopia; validate 1: Nigeria; test 2: USA-Mississippi, Brazil-Amazon) **and** temporal (features 2014–2019, label 2020–2024) — no spatial or temporal leakage.
- **Models trained & compared:** Logistic Regression (baseline), Random Forest, XGBoost.

**Verified test metrics (geographic + temporal holdout):**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.780 | 0.952 | 1.000 | 0.093 | 0.170 | 0.622 |
| **Random Forest (selected)** | **0.832** | **0.966** | 0.969 | 0.647 | **0.776** | 0.204 |
| XGBoost | 0.809 | 0.962 | 0.970 | 0.591 | 0.734 | 0.249 |

Confusion matrix (RF, test): [[326, 43], [722, 1325]].

- **Explainability:** SHAP TreeExplainer; top drivers: `early_flood_months`, `lon`, `early_n_detections`, `soil_moisture_std`, `early_flood_years` (full ranking in `ml/reports/model_report_v2.json`).
- **Honesty note (v1 → v2):** a first version (v1) reached ROC-AUC 0.99, but a self-audit found label leakage (`flood_years`/`n_detections` were derived from the same detection history that defined the label). v2 removed the leakage and re-framed the task temporally. Both reports are included (`model_report.json` = v1, `model_report_v2.json` = v2) as documentation of the audit.
- **Scope note:** this is **long-term flood susceptibility**, *not* a 24-hour flood probability forecast. No such claim is made anywhere.

### 1.2 Satellite water segmentation (U-Net, deep learning) — OPERATIONAL (subset-scale)
- **Data:** [isp-uv-es/WorldFloodsv2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2) (CC-BY-NC-4.0) — Sentinel-2 scenes + Copernicus EMS flood/water polygon masks, from *Global flood extent segmentation in optical satellite images*, Scientific Reports 13, 20316 (2023). Subset used: 15 train / 4 val / 4 test scenes from distinct EMS activations and countries (event-level split).
- **Model:** compact U-Net (483,025 params, 5 input bands: B, G, R, NIR, SWIR1), weighted BCE + Dice loss, flip/rotation augmentation, 256×256 patches, 12 epochs, CPU-only.
- **Verified test metrics (patch-level):** IoU 0.569, Dice 0.725, Precision 0.623, Recall 0.869, F1 0.725.
- **Per-scene IoU:** Grile (Italy) 0.913, Rattaloro 0.005, Porto Palo 0.214, EMSR438 Zimbabwe 0.408 — high variance across events is real and documented (see `ml/reports/unet_report.json` and visual outputs in `ml/reports/segmentation_visuals/`).

### 1.3 Honest limitations (documented, not hidden)
- **Susceptibility negatives:** the AI4G product contains only flood detections; "non-recurrent" cells are cells with few detections in the sample, **not** verified flood-free ground truth. Metrics are therefore optimistic for a true screening setting.
- **Segmentation subset:** training used 15 of 475 available train scenes (CPU budget); per-scene generalization varies strongly (see Rattaloro). Masks include permanent water (per WorldFloods convention), so "water fraction" ≠ "flood fraction" without change detection.
- **WorldFloods license is CC-BY-NC-4.0** — non-commercial use only.
- Class imbalance is high (83% positive cells); recall on the minority (non-recurrent) class is the weak spot of all three models.

---

## 2. Phase 2 — inference layer built on the saved artifacts

### 2.1 Alert engine (rule-based, calibrated)
- **Risk tiers** calibrated on the validation tile (N09E006, Nigeria): `high` = validation q90 (p ≥ 0.828), `moderate-high` = q75 (p ≥ 0.654), `moderate` = q50 (p ≥ 0.497).
- **Confirmed generalization:** test AUC on the held-out USA/Brazil tiles re-computed at inference time = **0.829** (244 high-risk cells out of 2,416).
- **Explainable alerts:** every alert carries its top-3 drivers (feature × importance × deviation from training median) in plain language — see `api/alerts_sample.json` (60 alerts: 56 Amazon / 4 Mississippi) and `api/predictions_test.parquet` (all 2,416 test cells with risk_p + tier).
- Files: `api/risk_tiers.json`, `api/alerts_sample.json`, `api/predictions_test.parquet`.

### 2.2 FastAPI inference service (`api/main.py`)
| Endpoint | Method | Purpose | Verified |
|---|---|---|---|
| `/health` | GET | liveness + model list | ✅ |
| `/models` | GET | model cards with test metrics | ✅ |
| `/susceptibility` | POST | RF risk_p + risk_tier per grid cell (≤2000 cells) | ✅ matches saved predictions exactly |
| `/segmentation` | POST | U-Net water probability on a HxWx5 S2 patch | ✅ real Grile test patch: predicted water 0.374 vs true 0.364 |

Run: `uvicorn api.main:app --port 8000` from the package root (loads both artifacts at startup; both models are loaded from `ml/models/`). The U-Net definition in the API is architecture-identical to `scripts/train_unet.py` (Block/dec naming, no extra normalization — reflectance/10000 clipped to [0,1]).

### 2.3 PostGIS schema (`db/schema.sql`)
PostgreSQL 15 + PostGIS 3.4 tables: `regions`, `model_versions` (seeded with both models + their verified metrics), `observations`, `predictions` (GiST-indexed cell geometries), `alerts` (severity watch/warning/critical, status lifecycle), `app_users`, `alert_acknowledgements`. Includes seed rows for the three known tiles and an alert-generation rule mirroring the calibrated threshold.

### 2.4 Dashboard (`dashboard/index.html`)
Single **self-contained** HTML file (no server, no CDN, ~130 KB) with the real test-tile predictions embedded (1,567 cells shown, all 244 high-risk): dark navy/cyan design system, KPI cards (AUC 0.832, PR-AUC 0.966, F1 0.776, Brier 0.204), canvas map with tier-colored cells sized by probability, hover tooltips, click-for-detail panel, region filter, and a 10-alert explainable feed (top-6 per test tile). Open directly in any browser.

**Honest scope note:** the dashboard is a static deliverable (this environment does not build the full React app); the production frontend remains a planned phase, and this file is its working prototype with real data.

---

## 3. Reproduce everything

```bash
pip install pandas pyarrow scikit-learn xgboost shap torch rasterio matplotlib joblib
python scripts/download_ai4g.py          # fetch AI4G tiles from HuggingFace
python scripts/sample_ai4g.py            # stream-sample to ~6M rows
python scripts/train_flood_susceptibility_v2.py   # EDA + 3 models + SHAP + report
python scripts/download_worldfloods.py   # fetch S2 scenes + EMS floodmaps
python scripts/train_unet.py             # U-Net training (resumable) + eval + visuals
```

Every script is deterministic (fixed seeds) and writes its own report JSON + PNG figures.

## 4. Files in this package

```
disaster-x-ai/
├── README.md                    ← this file
├── api/                         ← Phase 2+3: inference layer
│   ├── main.py                  (FastAPI: /health /models /susceptibility /segmentation
│   │                             /segmentation/geotiff /predictions /alerts
│   │                             /analytics/features /landslide/fatality — 9 verified endpoints)
│   ├── requirements.txt         (API dependencies for Docker)
│   ├── risk_tiers.json          (calibrated thresholds + validation coverage)
│   ├── alerts_sample.json       (60 explainable alerts on held-out test tiles)
│   └── predictions_test.parquet (2,416 test cells: risk_p, risk_tier, label)
├── db/
│   ├── schema.sql               (PostgreSQL 15 + PostGIS 3.4 schema, seeded)
│   ├── alembic.ini / env.py     (Alembic configuration — DATABASE_URL aware)
│   └── versions/0001_initial.py (validated migration: upgrade/seed/roundtrip/downgrade)
├── docker/
│   ├── Dockerfile.api           (Python 3.12 + models + API)
│   ├── Dockerfile.frontend      (node build stage → nginx serve)
│   ├── nginx.conf               (SPA + /api reverse proxy)
│   └── docker-compose.yml       (db + api + frontend, healthchecked)
├── frontend/                    ← Phase 3: React 19 + TS + Tailwind 4 + Leaflet + recharts
│   └── src/…                    (npm run build verified; live E2E tested in browser)
├── dashboard/
│   └── index.html               (self-contained static dashboard, open in any browser)
├── docs/                        ← Phase 3 documentation set
│   ├── DISASTER-X_AI_Final_Report.docx / .pdf   (10 pages, TOC, figures, tables)
│   ├── DISASTER-X_AI_IEEE_Draft.docx / .pdf     (paper draft with references)
│   └── DISASTER-X_AI_Viva_QA.docx / .pdf        (15 anticipated questions with answers)
├── ml/
│   ├── models/                  ← trained artifacts (loadable, independent of app code)
│   │   ├── flood_susceptibility_v2_rf.joblib   (selected model + feature list + split metadata)
│   │   ├── flood_susceptibility_rf.joblib      (v1, kept for audit trail)
│   │   ├── unet_worldfloods.pt                  (U-Net weights)
│   │   ├── landslide_fatality_gb.joblib         (GLC fatality screen)
│   │   └── landslide_unet_l4s.pt                (Landslide4Sense U-Net, experimental)
│   ├── reports/                 ← verified metrics, EDA, SHAP, training curves
│   │   ├── model_report_v2.json / model_report.json
│   │   ├── unet_report.json
│   │   ├── landslide_glc_report.json / landslide_glc_eda.png
│   │   ├── landslide_unet_report.json / landslide_unet_curve.png
│   │   ├── landslide_segmentation_visuals/
│   │   ├── eda_*.json / eda_*.png / model_comparison*.png / shap_summary*.png
│   │   ├── unet_training_curve.png
│   │   └── segmentation_visuals/   (original | ground truth | prediction | overlay per test scene)
│   └── data/                    ← processed datasets used for training/eval
│       ├── ai4g_sample.parquet  (~6M detections, 11 tiles)
│       ├── grid_dataset.parquet (v1 grid cells)
│       └── susceptibility_dataset_v2.parquet (v2 leakage-free modeling table)
└── scripts/                     ← all pipeline code (download → sample → train → evaluate)
```

## 4bis. Phase 3 — landslide module, extended API, frontend, deployment, docs

### 4bis.1 Landslide module (new)
- **Event fatality screening (operational):** logistic pipeline over country/trigger/category/setting/month/location on the NASA Global Landslide Catalog (11,337 events, 1988–2018; temporal split train ≤2014 / val 2015–16 / test ≥2017). Test ROC-AUC **0.886**, PR-AUC 0.575, Brier 0.094 at a 15.2% base rate. Served at `POST /landslide/fatality` with the not-a-hazard-model caveat in the response.
- **Landslide segmentation (experimental):** two-level U-Net on 7 bands (B,G,R,NIR,SWIR1,slope,DEM) of Landslide4Sense, trained on **260 of 3,799** official train patches (CPU budget), 40-patch local test. Image IoU **0.146**, Dice 0.211 — published as an explicitly subset-scale baseline, not a production capability.
- Files: `ml/models/landslide_fatality_gb.joblib`, `ml/models/landslide_unet_l4s.pt`, `ml/reports/landslide_*`.

### 4bis.2 Extended API (v3.0.0)
Nine verified endpoints (all tested in-process on real data): the Phase 2 set plus `POST /segmentation/geotiff` (multipart upload, tiled inference with reflect-padding for non-multiple-of-8 edges; real Grile scene: predicted water 0.3485 vs true 0.3500), `GET /predictions` (paged, tier filter), `GET /alerts` (tile filter), `GET /analytics/features`, and `POST /landslide/fatality`. CORS enabled for frontend origins. Two bugs were found and fixed during verification (U-Net layer naming vs checkpoint; edge-tile padding).

### 4bis.3 React frontend (source tree)
`frontend/` — React 19 + TypeScript + Tailwind 4 + react-leaflet + recharts. `npm run build` passes with strict TypeScript; the built app was served behind an `/api` proxy and verified in a real headless browser: KPI cards (0.832 / 0.966 / 0.776 / 244), 244 tier-colored map markers, tier-distribution + feature-importance charts, 20-alert explainable feed, detail modal with drivers, tier filtering. A real crash (empty-array NaN center on slow loads) was caught by this test and fixed.

### 4bis.4 Deployment artifacts
`docker/docker-compose.yml` (postgis/postgis:16-3.4 + api + nginx frontend, healthchecks, schema auto-init), `Dockerfile.api`, `Dockerfile.frontend` (multi-stage node→nginx), `nginx.conf` (SPA + `/api` proxy + upload size). **Honesty note:** Docker was not available in the build environment, so the stack is delivered ready-to-build but not run-validated as a composition; each component (API, frontend build, schema, migration) was validated directly.

### 4bis.5 Alembic migrations
`db/alembic.ini` + `db/env.py` (DATABASE_URL-aware) + `db/versions/0001_initial.py`. Validated end-to-end on a live database engine: upgrade creates all 7 tables + indexes + seeds the model registry, insert/query roundtrip succeeds, downgrade is clean. PostGIS-specific DDL is guarded to PostgreSQL.

### 4bis.6 Documentation set (docs/)
- **Final Project Report** (10 pages): executive summary, data & methodology, all four models with verified metrics, platform layer, honest limitations, roadmap. Cover (R1/DM-1), TOC, 4 tables, 3 figures.
- **IEEE-style paper draft** (4 pages): abstract, related work, data, methods, results tables, and a section on the leakage audit as a finding, with 4 references.
- **Viva Q&A preparation** (4 pages): 15 anticipated defense questions with model answers, key numbers to memorize, and traps to avoid.

## 5. Status of the full platform & next steps

Delivered: the **AI core** (Phase 1), the **inference layer** (Phase 2), and the **completion phase** (Phase 3: landslide module, extended API, React frontend source, Docker orchestration, Alembic migrations, documentation set) — all on real data with verified metrics.

Remaining work (needs an environment this package could not provide):

1. **Live deployment**: build and run the Compose stack on a Docker host; apply the migration to a real PostGIS instance and wire the API to it (DATABASE_URL is already supported).
2. **Scale training on GPU**: full WorldFloods (475 scenes) and Landslide4Sense (3,799 patches) training runs — both pipelines are input-complete and only compute-bound.
3. **Near-term forecasting**: join numerical weather prediction outputs with the susceptibility scores (the schema's `observations` table already anticipates external feeds).
4. **Auth & multi-tenancy** on the existing `app_users` schema; alert severity escalation rules calibrated per region.

Send a follow-up message to continue from these saved artifacts.
