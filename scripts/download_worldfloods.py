"""Download a CPU-scale subset of WorldFloods v2 (Sentinel-2 + flood masks).

Picks the smallest train/val/test scenes so a 2-core CPU can train a U-Net
on 256x256 patches in reasonable time. Band layout (WorldFloods v2 S2):
bands 1-4 = S2 L1C RGBN, 8-13 = S2 L2A SWIR etc; masks: 0 land, 1 water,
2 invalid/border, 3 interpreter disagreement. We use mask==1 as flood/water.
"""
import os
import requests

BASE = "https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/resolve/main"
OUT = "/home/z/my-project/ml/data/worldfloods"
os.makedirs(OUT, exist_ok=True)

# small scenes (< ~70 MB) from different events/countries
PICKS = {
    "train": [
        "EMSR249_03GALWAY_DEL_v2.tif",
        "EMSR249_07KILRUSHTARBERT_DEL_v2.tif",
        "EMSR249_14WATERVILLE_DEL_v2.tif",
        "EMSR249_15LAHINCH_DEL_v1.tif",
        "EMSR255_03SCHONECK_DEL_v1.tif",
        "EMSR258_01DIVJAKE_DEL_v2.tif",
        "EMSR258_03KOLONJE_DEL_v2.tif",
        "EMSR258_05PATOS_DEL_v3.tif",
        "01042016_Holmes_Creek_at_Vernon_FL.tif",
        "EMSR312_02CALAYAN_GRA_v1.tif",
        "EMSR478_AOI01_GRA_PRODUCT.tif",
        "EMSR395_AOI02_GRA_PRODUCT.tif",
        "EMSR431_AOI07_GRA_PRODUCT.tif",
        "EMSR485_AOI03_GRA_PRODUCT.tif",
        "EMSR470_AOI04_GRA_PRODUCT.tif",
        "EMSR357_AOI06_GRA_PRODUCT.tif",
    ],
    "val": [
        "EMSR442_AOI03_DEL_PRODUCT.tif",
        "EMSR358_AOI01_FEP_PRODUCT.tif",
        "EMSR329_05CASTIADAS_GRA_v3.tif",
        "EMSR260_07CAMPOGALLIANO_DEL_MONIT01_v1.tif",
    ],
    "test": [
        "EMSR273_01GRILE_DEL_MONIT04_v2.tif",
        "EMSR438_AOI06_DEL_PRODUCT.tif",
        "EMSR333_01RATTALORO_DEL_MONIT01_v1.tif",
        "EMSR333_02PORTOPALO_DEL_MONIT01_v1.tif",
    ],
}
# mask counterparts live under train/floodmaps etc.
for split in ["train", "val", "test"]:
    files = PICKS[split]
    listing = requests.get(f"https://huggingface.co/api/datasets/isp-uv-es/WorldFloodsv2/tree/main/{split}/S2", timeout=120).json()
    sizes = {x["path"].split("/")[-1]: x["size"] for x in listing}
    masks = requests.get(f"https://huggingface.co/api/datasets/isp-uv-es/WorldFloodsv2/tree/main/{split}/floodmaps", timeout=120).json()
    mask_names = {x["path"].split("/")[-1] for x in masks}
    os.makedirs(f"{OUT}/{split}/S2", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/floodmaps", exist_ok=True)
    for f in files:
        # image
        dest = f"{OUT}/{split}/S2/{f}"
        if not os.path.exists(dest):
            url = f"{BASE}/{split}/S2/{f}"
            with requests.get(url, timeout=600, stream=True) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.rename(dest + ".part", dest)
        print(split, "img", f, round(sizes.get(f, 0)/1e6, 1), "MB")
        # mask: find matching name (floodmaps may have slightly different suffixes)
        stem = f.rsplit("_", 1)[0]
        cands = [m for m in mask_names if m.startswith(f)] + [m for m in mask_names if m.startswith(stem)]
        if cands:
            m = cands[0]
            mdest = f"{OUT}/{split}/floodmaps/{m}"
            if not os.path.exists(mdest):
                url = f"{BASE}/{split}/floodmaps/{m}"
                with requests.get(url, timeout=600, stream=True) as r:
                    r.raise_for_status()
                    with open(mdest + ".part", "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.rename(mdest + ".part", mdest)
            print(split, "mask", m)
        else:
            print(split, "NO MASK FOR", f)
print("done")
