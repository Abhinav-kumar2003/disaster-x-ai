"""DISASTER-X AI — FastAPI inference service (Phases 2–3).

Serves the verified models and production read paths:
  GET  /health /models
  POST /susceptibility            – flood susceptibility RF (grid cells)
  POST /segmentation              – U-Net water segmentation (S2 patch)
  POST /segmentation/geotiff      – U-Net on an uploaded GeoTIFF (multipart)
  GET  /predictions               – precomputed test-tile predictions (paged)
  GET  /alerts                    – alert feed from the calibrated engine
  GET  /analytics/features        – feature importances + tier distribution
  POST /landslide/fatality        – GLC event fatality screening model

Run:  uvicorn api.main:app --port 8000   (from the disaster-x-ai/ root)
Docker: see docker/Dockerfile.api
"""
from __future__ import annotations

import io
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parents[1]
TIER_THR = (0.8278, 0.6541, 0.4968)  # calibrated on validation tile N09E006


def _tier(p: float) -> str:
    if p >= TIER_THR[0]:
        return "high"
    if p >= TIER_THR[1]:
        return "moderate-high"
    if p >= TIER_THR[2]:
        return "moderate"
    return "low"


app = FastAPI(
    title="DISASTER-X AI inference API",
    description="Flood susceptibility (RF) + water segmentation (U-Net) + landslide fatality screening on real trained artifacts.",
    version="3.0.0",
)

# CORS for the bundled dashboard / future frontend (same-origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- susceptibility ----------
_art = joblib.load(ROOT / "ml/models/flood_susceptibility_v2_rf.joblib")
RF, FEATURES = _art["model"], _art["features"]

# precomputed test-tile predictions + alerts (produced by the calibration run)
_PREDS = pd.read_parquet(ROOT / "api/predictions_test.parquet")
_ALERTS = json.loads((ROOT / "api/alerts_sample.json").read_text())

MODEL_CARDS = {
    "susceptibility_rf_v2": {
        "task": "long-term flood susceptibility: P(>=2 distinct Sentinel-1 flood months in 2020-2024) from 2014-2019 evidence + static features",
        "test_auc": 0.832,
        "features": FEATURES,
        "calibrated_on": "validation tile N09E006 (Nigeria)",
        "artifact": "ml/models/flood_susceptibility_v2_rf.joblib",
    },
    "segmentation_unet": {
        "task": "water segmentation on Sentinel-2 patches (B,G,R,NIR,SWIR1), 256x256",
        "test_iou": 0.569,
        "params": 483025,
        "artifact": "ml/models/unet_worldfloods.pt",
    },
    "landslide_fatality_lr": {
        "task": "screening: P(landslide event is fatal) from country/trigger/category/setting/month/location",
        "test_auc": 0.886,
        "artifact": "ml/models/landslide_fatality_gb.joblib",
    },
}


class CellFeatures(BaseModel):
    early_flood_months: float
    early_flood_years: float
    early_n_detections: float
    dem_metric_1: float
    dem_metric_2: float
    dem_metric_1_max: float
    soil_moisture: float
    soil_moisture_std: float
    soil_moisture_zscore: float
    temp: float
    land_cover: float
    dominant_month: float
    lat: float
    lon: float


class SusceptibilityRequest(BaseModel):
    cells: list[CellFeatures] = Field(min_length=1, max_length=2000)


class SusceptibilityItem(BaseModel):
    risk_p: float
    risk_tier: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": list(MODEL_CARDS)}


@app.get("/models")
def models() -> dict:
    return MODEL_CARDS


@app.post("/susceptibility")
def susceptibility(req: SusceptibilityRequest) -> dict:
    X = pd.DataFrame([c.model_dump() for c in req.cells])[FEATURES]
    p = RF.predict_proba(X)[:, 1]
    return {
        "predictions": [
            SusceptibilityItem(risk_p=round(float(pi), 4), risk_tier=_tier(float(pi)))
            for pi in p
        ]
    }


# ---------- production read paths ----------
@app.get("/predictions")
def predictions(
    tier: str | None = Query(None, pattern="^(low|moderate|moderate-high|high)$"),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> dict:
    df = _PREDS
    if tier:
        df = df[df.risk_tier == tier]
    page = df.iloc[offset : offset + limit]
    return {
        "total": int(len(df)),
        "offset": offset,
        "limit": limit,
        "predictions": [
            {
                "lat": float(r.lat), "lon": float(r.lon),
                "risk_p": float(r.risk_p), "risk_tier": r.risk_tier,
                "tile": r.tile_id, "label": int(r.label),
            }
            for r in page.itertuples()
        ],
    }


@app.get("/alerts")
def alerts(
    tile: str | None = Query(None, pattern="^(N33W090|S09W060)$"),
    limit: int = Query(20, ge=1, le=60),
) -> dict:
    feed = [a for a in _ALERTS if tile is None or a["tile"] == tile][:limit]
    return {"total": len(feed), "alerts": feed}


@app.get("/analytics/features")
def analytics_features() -> dict:
    imp = RF.feature_importances_
    order = np.argsort(-imp)[:10]
    tiers = _PREDS.risk_tier.value_counts().to_dict()
    return {
        "top_features": [
            {"feature": FEATURES[i], "importance": round(float(imp[i]), 4)}
            for i in order
        ],
        "tier_distribution": tiers,
        "n_cells": int(len(_PREDS)),
        "tier_thresholds": {
            "high": TIER_THR[0], "moderate-high": TIER_THR[1],
            "moderate": TIER_THR[2],
        },
    }


# ---------- landslide fatality screening ----------
_LS = joblib.load(ROOT / "ml/models/landslide_fatality_gb.joblib")
LS_MODEL, LS_FEATS = _LS["model"], _LS["features"]


class LandslideEvent(BaseModel):
    country: str = Field(examples=["India"])
    ls_trig: str = Field(examples=["downpour"])
    ls_cat: str = Field(examples=["landslide"])
    ls_setting: str = Field(examples=["slope"])
    month: int = Field(ge=1, le=12)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@app.post("/landslide/fatality")
def landslide_fatality(ev: LandslideEvent) -> dict:
    X = pd.DataFrame([ev.model_dump()])[LS_FEATS]
    p = float(LS_MODEL.predict_proba(X)[0, 1])
    return {
        "fatal_p": round(p, 4),
        "screen": "high" if p >= 0.5 else "low",
        "note": "screening model on reporting+trigger features; not a hazard model",
    }


# ---------- segmentation ----------
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class Block(nn.Module):
    """Conv-BN-ReLU x2 block — identical to scripts/train_unet.py."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Compact U-Net — identical architecture to scripts/train_unet.py."""

    def __init__(self, cin: int = 5, base: int = 16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.d1 = Block(cin, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.bottleneck = Block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = Block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = Block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = Block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        bn = self.bottleneck(self.pool(s3))
        y = self.dec3(torch.cat([self.u3(bn), s3], dim=1))
        y = self.dec2(torch.cat([self.u2(y), s2], dim=1))
        y = self.dec1(torch.cat([self.u1(y), s1], dim=1))
        return self.out(y)


_UNET = UNet()
_UNET.load_state_dict(
    torch.load(ROOT / "ml/models/unet_worldfloods.pt", map_location="cpu")
)
_UNET.eval()
# training normalization: reflectance / 10000, clipped to [0,1] — nothing more


class SegmentationRequest(BaseModel):
    patch: list[list[list[float]]] = Field(
        description="256x256x5 normalized reflectance [B,G,R,NIR,SWIR1] in [0,1]"
    )


@app.post("/segmentation")
def segmentation(req: SegmentationRequest) -> dict:
    try:
        arr = torch.tensor(req.patch, dtype=torch.float32)
        if arr.ndim != 3 or arr.shape[-1] != 5 or arr.shape[0] < 32:
            raise ValueError(
                f"expected HxWx5 patch, got {tuple(arr.shape)} (H,W >= 32)"
            )
    except Exception as e:
        raise HTTPException(400, f"invalid patch: {e}") from e
    with torch.no_grad():
        logit = _UNET(arr.permute(2, 0, 1).unsqueeze(0))
        prob = torch.sigmoid(logit)[0, 0]
    return {
        "shape": [int(prob.shape[0]), int(prob.shape[1])],
        "water_fraction": round(float((prob > 0.5).float().mean()), 4),
        "probability_min": round(float(prob.min()), 4),
        "probability_max": round(float(prob.max()), 4),
        "probability_mean": round(float(prob.mean()), 4),
    }


@app.post("/segmentation/geotiff")
async def segmentation_geotiff(file: UploadFile = File(...)) -> dict:
    """U-Net on an uploaded Sentinel-2 GeoTIFF (bands B,G,R,NIR,SWIR1).

    Processes the scene in 256x256 tiles and returns the overall water
    fraction plus per-tile stats.
    """
    try:
        import rasterio

        data = await file.read()
        with rasterio.open(io.BytesIO(data)) as ds:
            # model bands: S2 B,G,R,NIR,SWIR1 = file channels 1,2,3,8,11
            # (1-indexed), exactly as in scripts/train_unet.py BANDS
            band_idx = [1, 2, 3, 8, 11] if ds.count >= 11 else list(
                range(1, min(5, ds.count) + 1)
            )
            arr = ds.read(band_idx).astype(np.float32) / 10000.0
            arr = np.clip(arr, 0, 1)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"cannot read GeoTIFF: {e}") from e

    _, h, w = arr.shape
    if h < 32 or w < 32:
        raise HTTPException(400, "image too small (need >=32x32)")
    tiles, water_px, total_px = [], 0, 0
    with torch.no_grad():
        for y0 in range(0, h - 31, 256):
            for x0 in range(0, w - 31, 256):
                tile = arr[:, y0 : y0 + 256, x0 : x0 + 256]
                th, tw = tile.shape[1], tile.shape[2]
                if th < 32 or tw < 32:
                    continue
                # U-Net needs dims divisible by 8 (3 pooling levels): pad, crop back
                ph = (8 - th % 8) % 8
                pw = (8 - tw % 8) % 8
                t = torch.from_numpy(tile).unsqueeze(0)
                if ph or pw:
                    t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
                logit = _UNET(t)
                prob = torch.sigmoid(logit)[0, 0, :th, :tw]
                wf = float((prob > 0.5).float().mean())
                tiles.append(
                    {"y": y0, "x": x0, "h": th, "w": tw,
                     "water_fraction": round(wf, 4)}
                )
                water_px += int((prob > 0.5).sum())
                total_px += th * tw
    return {
        "image_size": [h, w],
        "n_tiles": len(tiles),
        "water_fraction": round(water_px / max(total_px, 1), 4),
        "tiles": tiles[:200],
    }
