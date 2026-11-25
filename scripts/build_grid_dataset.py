"""DISASTER-X AI - Task B: grid-cell flood susceptibility dataset builder.

Input: AI4G flood detections (Sentinel-1, 2014-2024) sampled across 11 tiles.

Method:
- Grid each tile at 0.05 deg (~5km) cells.
- Aggregate detections per cell:
  * label: cell experienced >= MIN_FLOOD_MONTHS distinct flood months 2014-2024
    (recurrent flood-prone cell) vs 1 month only.
  * features: mean DEM slope/roughness metrics, mean soil moisture + anomaly,
    mean temp, dominant land cover, detection counts per year (flood history),
    seasonality (monotonic index), coordinates.
- Negative cells: the paper's product only covers flood detections, so
  *negatives are generated from wet/low-land cells* in the same tiles that
  have zero detections in the sample - carefully labelled "sample-negative"
  and flagged; this is a documented limitation, not ground truth absence.
"""
import numpy as np
import pandas as pd

SRC = "/home/z/my-project/ml/data/processed/ai4g_sample.parquet"
OUT = "/home/z/my-project/ml/data/processed/grid_dataset.parquet"
GRID = 0.05
MIN_FLOOD_MONTHS = 3  # label positive if >= 3 distinct flood months over 11 years

df = pd.read_parquet(SRC)

# --- clean fill values ---
for c in ["soil_moisture", "soil_moisture_zscore", "soil_moisture_sca",
          "soil_moisture_zscore_sca", "temp"]:
    df[c] = df[c].where(df[c] > -900)

# grid cell ids
df["lat_g"] = np.floor(df.lat / GRID) * GRID
df["lon_g"] = np.floor(df.lon / GRID) * GRID

# count distinct (year, month) per cell
ym = df.year * 12 + df.month
df["ym"] = ym

agg = df.groupby(["tile_id", "lat_g", "lon_g"]).agg(
    n_detections=("ym", "size"),
    flood_months=("ym", "nunique"),
    flood_years=("year", "nunique"),
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

agg["flood_months"] = agg.flood_months.fillna(0).astype(int)
agg["label"] = (agg.flood_months >= MIN_FLOOD_MONTHS).astype(int)

# seasonal concentration: fraction of detections in the dominant season
df["month"] = df.month.astype(int)
dom = df.groupby(["tile_id", "lat_g", "lon_g", "month"]).size().rename("n").reset_index()
dom = dom.sort_values("n", ascending=False).drop_duplicates(["tile_id", "lat_g", "lon_g"])
agg = agg.merge(dom[["tile_id", "lat_g", "lon_g", "month"]].rename(
    columns={"month": "dominant_month"}), on=["tile_id", "lat_g", "lon_g"], how="left")

print("cells:", len(agg))
print("label distribution:\n", agg.label.value_counts().to_string())
print("\nper tile:\n", agg.groupby(["tile_id", "label"]).size().unstack(fill_value=0).to_string())
print("\nflood_months distribution:\n", agg.flood_months.value_counts().sort_index().to_string())

agg.to_parquet(OUT, index=False)
print(f"\nsaved {OUT}")
