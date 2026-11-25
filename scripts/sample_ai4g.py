"""Memory-safe streaming sampler for AI4G tiles.

Reads each parquet in row batches (pyarrow), keeps rows with probability
p = min(1, TARGET/total_rows) so memory stays bounded, aggregates later.
"""
import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = "/home/z/my-project/ml/data/raw/ai4g"
OUT = "/home/z/my-project/ml/data/processed"
TARGET = 600_000  # rows per tile max
rng = np.random.default_rng(42)

os.makedirs(OUT, exist_ok=True)
frames = []
for path in sorted(glob.glob(f"{RAW}/*.parquet")):
    tile = os.path.basename(path).split("-")[0]
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    p = min(1.0, TARGET / total)
    kept = []
    for batch in pf.iter_batches(batch_size=500_000):
        bdf = batch.to_pandas()
        if p < 1.0:
            mask = rng.random(len(bdf)) < p
            bdf = bdf[mask]
        if len(bdf):
            kept.append(bdf)
    df = pd.concat(kept, ignore_index=True)
    df["tile_id"] = tile
    yrs = df.groupby(["year"]).size()
    print(
        f"{tile}: raw={total:,} kept={len(df):,} years={df.year.min()}-{df.year.max()} "
        f"fp_rate={df.edge_false_positives.mean():.3f} unique_cells={df.groupby(['lat','lon']).ngroups:,}",
        flush=True,
    )
    frames.append(df)
    del kept, df

out = pd.concat(frames, ignore_index=True)
out.to_parquet(f"{OUT}/ai4g_sample.parquet", index=False)
print(f"\ncombined {len(out):,} rows -> {OUT}/ai4g_sample.parquet", flush=True)
