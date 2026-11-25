"""Download AI4G flood detection tiles (parquet only) from HuggingFace.

Selection: geographic spread across continents / climates, moderate size.
Each tile = 3 deg x 3 deg of Sentinel-1 flood detections 2014-2024 with
auxiliary environmental variables (DEM metrics, soil moisture, land cover,
temperature). Tiles are also the natural geographic split unit for ML.
"""
import os
import time
import requests

BASE = "https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset/resolve/main"
OUT = "/home/z/my-project/ml/data/raw/ai4g"
os.makedirs(OUT, exist_ok=True)

# Diverse tiles: Africa (Ethiopia, Somalia, Nigeria, Sudan, Mozambique),
# South Asia (Bangladesh, India), SE Asia (Vietnam, Cambodia), South America (Brazil),
# North America (USA), Australia.
TILES = [
    "N03/N03E042",   # Ethiopia/Somalia border (paper example, Dolo Ado)
    "N24/N24E090",   # Bangladesh (Brahmaputra/Ganges delta)
    "N09/N09E006",   # Nigeria (Niger river)
    "N15/N15E033",   # Sudan (Nile)
    "S18/S18E033",   # Mozambique (Zambezi)
    "N12/N12E105",   # Cambodia (Tonle Sap / Mekong)
    "S09/S09W060",   # Brazil (Amazon)
    "N33/N33W090",   # USA (Mississippi)
    "S30/S30E147",   # Australia (Murray-Darling)
    "N27/N27E069",   # Pakistan (Indus)
    "N21/N21E078",   # India (central)
]

for tile in TILES:
    name = tile.split("/")[-1]
    dest = os.path.join(OUT, f"{name}-post-processing.parquet")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"skip {name} (exists)")
        continue
    url = f"{BASE}/{tile}/{name}-post-processing.parquet"
    t0 = time.time()
    try:
        with requests.get(url, timeout=600, stream=True) as r:
            r.raise_for_status()
            with open(dest + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.rename(dest + ".part", dest)
        print(f"OK {name} {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"FAIL {name}: {e}")
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
