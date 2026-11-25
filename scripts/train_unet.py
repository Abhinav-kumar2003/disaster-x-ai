"""DISASTER-X AI - U-Net water segmentation training (WorldFloods v2 subset).

- Data: Sentinel-2 L1C scenes + Copernicus EMS water/flood polygon masks
  (rasterized to the image grid). 15 train / 4 val / 4 test scenes,
  geographic event-level split (different EMSR activations & countries).
- Input: bands B2 (blue), B3 (green), B4 (red), B8 (NIR), B11 (SWIR1) as
  channels 1,2,3,8,11 of the GeoTIFF; percentile-normalized to [0,1].
- Model: compact U-Net (PyTorch, CPU), 256x256 patches, weighted BCE+Dice loss
  to handle class imbalance, flip/rot90 augmentation.
- Metrics: IoU, Dice, precision, recall, pixel F1 on held-out scenes.
- Saves best checkpoint + metrics report + prediction visualizations.
"""
import glob
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = "/home/z/my-project/ml/data/worldfloods"
OUT = "/home/z/my-project/ml/reports"
ART = "/home/z/my-project/ml/models"
BANDS = [1, 2, 3, 8, 11]  # B G R NIR SWIR1 (1-indexed)
PATCH = 256
BATCH = 8
EPOCHS = 12
LR = 2e-4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


# ---------------- data ----------------
def scene_stats(path):
    with rasterio.open(path) as ds:
        arr = ds.read(BANDS).astype(np.float32)
    arr = arr / 10000.0
    arr = np.clip(arr, 0, 1)
    return arr


class PatchDataset(Dataset):
    """Patches sampled from scenes, water-balanced when possible."""

    def __init__(self, split, augment=False, max_patches_per_scene=40, seed=SEED):
        self.items = []
        rng = np.random.default_rng(seed)
        for img_path in sorted(glob.glob(f"{ROOT}/{split}/S2/*.tif")):
            name = os.path.basename(img_path).replace(".tif", "")
            mask_path = f"{ROOT}/{split}/masks/{name}.tif"
            if not os.path.exists(mask_path):
                continue
            img = scene_stats(img_path)
            with rasterio.open(mask_path) as ds:
                mask = ds.read(1).astype(np.float32)
            if img.shape[1:] != mask.shape:
                continue
            h, w = mask.shape
            n = 0
            attempts = 0
            while n < max_patches_per_scene and attempts < max_patches_per_scene * 6:
                attempts += 1
                i = rng.integers(0, max(h - PATCH, 1))
                j = rng.integers(0, max(w - PATCH, 1))
                if h < PATCH or w < PATCH:
                    ph = np.zeros((len(BANDS), PATCH, PATCH), np.float32)
                    pm = np.zeros((PATCH, PATCH), np.float32)
                    hh, ww = min(h, PATCH), min(w, PATCH)
                    ph[:, :hh, :ww] = img[:, :hh, :ww]
                    pm[:hh, :ww] = mask[:hh, :ww]
                    xi, mi = ph, pm
                else:
                    xi = img[:, i:i + PATCH, j:j + PATCH]
                    mi = mask[i:i + PATCH, j:j + PATCH]
                # keep patches with some water half the time, any patch otherwise
                if mi.mean() > 0.005 or rng.random() < 0.5:
                    self.items.append((xi.astype(np.float32), mi.astype(np.float32)))
                    n += 1
        self.augment = augment
        self.rng = np.random.default_rng(seed + 1)
        print(f"{split}: {len(self.items)} patches from {len(glob.glob(f'{ROOT}/{split}/masks/*.tif'))} scenes")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, m = self.items[idx]
        if self.augment:
            k = self.rng.integers(0, 4)
            x = np.rot90(x, k, axes=(1, 2)).copy()
            m = np.rot90(m, k, axes=(0, 1)).copy()
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1].copy()
                m = m[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(m[None])


# ---------------- U-Net ----------------
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=5, base=16):
        super().__init__()
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
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        bn = self.bottleneck(self.pool(s3))
        x = self.dec3(torch.cat([self.u3(bn), s3], dim=1))
        x = self.dec2(torch.cat([self.u2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.u1(x), s1], dim=1))
        return self.out(x)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def combined_loss(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight))
    return bce + dice_loss(logits, target)


# ---------------- training ----------------
def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"loss": 0.0, "n": 0}
    inter_sum, union_sum, tp, fp, fn = 0.0, 0.0, 0.0, 0.0, 0.0
    for x, m in loader:
        logits = model(x)
        pw = max(1.0, (m == 0).float().mean().item() / max((m == 1).float().mean().item(), 1e-6))
        loss = combined_loss(logits, m, pw)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        totals["loss"] += loss.item() * len(x)
        totals["n"] += len(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter_sum += (pred * m).sum().item()
        union_sum += ((pred + m) > 0).sum().item()
        tp += (pred * m).sum().item()
        fp += (pred * (1 - m)).sum().item()
        fn += ((1 - pred) * m).sum().item()
    iou = inter_sum / max(union_sum, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return totals["loss"] / max(totals["n"], 1), iou, dice, prec, rec, f1


def main():
    train_ds = PatchDataset("train", augment=True, max_patches_per_scene=30)
    val_ds = PatchDataset("val", augment=False, max_patches_per_scene=30)
    test_ds = PatchDataset("test", augment=False, max_patches_per_scene=60)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = UNet(cin=len(BANDS), base=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params:,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_iou = -1
    history = []
    ckpt = f"{ART}/unet_worldfloods.pt"
    resume = f"{ART}/unet_resume.pt"
    start_ep = 1
    # resume from previous chunk if present (sandbox kills long jobs)
    if os.path.exists(resume):
        state = torch.load(resume, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_val_iou = state["best_val_iou"]
        start_ep = state["epoch"] + 1
        print(f"resuming from epoch {start_ep}", flush=True)
    for ep in range(start_ep, EPOCHS + 1):
        tr = run_epoch(model, train_ld, optimizer)
        va = run_epoch(model, val_ld)
        scheduler.step()
        history.append({"epoch": ep, "train_loss": tr[0], "train_iou": tr[1],
                        "val_loss": va[0], "val_iou": va[1], "val_dice": va[2]})
        print(f"ep {ep:02d} train_loss={tr[0]:.4f} train_iou={tr[1]:.4f} | "
              f"val_loss={va[0]:.4f} val_iou={va[1]:.4f} val_dice={va[2]:.4f}", flush=True)
        if va[1] > best_val_iou:
            best_val_iou = va[1]
            torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "best_val_iou": best_val_iou, "epoch": ep}, resume)

    model.load_state_dict(torch.load(ckpt))
    te = run_epoch(model, test_ld)
    print(f"\nTEST: loss={te[0]:.4f} IoU={te[1]:.4f} Dice={te[2]:.4f} "
          f"P={te[3]:.4f} R={te[4]:.4f} F1={te[5]:.4f}")

    # ---------------- per-scene evaluation + visualization ----------------
    vis_dir = f"{OUT}/segmentation_visuals"
    os.makedirs(vis_dir, exist_ok=True)
    scene_metrics = []
    model.eval()
    for img_path in sorted(glob.glob(f"{ROOT}/test/S2/*.tif")):
        name = os.path.basename(img_path).replace(".tif", "")
        mask_path = f"{ROOT}/test/masks/{name}.tif"
        if not os.path.exists(mask_path):
            continue
        img = scene_stats(img_path)
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1).astype(np.float32)
        h, w = mask.shape
        # pad to multiple of 8 (ceiling)
        ph = (h + 7) // 8 * 8
        pw = (w + 7) // 8 * 8
        x = np.zeros((len(BANDS), ph, pw), np.float32)
        x[:, :h, :w] = img
        with torch.no_grad():
            logits = model(torch.from_numpy(x)[None])[0, 0, :h, :w].numpy()
        pred = (logits > 0).astype(np.float32)
        inter = (pred * mask).sum()
        union = ((pred + mask) > 0).sum()
        iou = inter / max(union, 1)
        tp = (pred * mask).sum(); fp = (pred * (1 - mask)).sum(); fn = ((1 - pred) * mask).sum()
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        scene_metrics.append({"scene": name, "iou": float(iou), "dice": float(dice),
                              "precision": float(prec), "recall": float(rec), "f1": float(f1),
                              "water_frac_true": float(mask.mean()), "water_frac_pred": float(pred.mean())})
        print(f"scene {name}: IoU={iou:.3f} Dice={dice:.3f} P={prec:.3f} R={rec:.3f}")
        # visualize: RGB, true mask, pred mask, overlay
        rgb = np.clip(np.moveaxis(img[[2, 1, 0]], 0, -1) * 3.2, 0, 1)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title("Sentinel-2 RGB", fontsize=9)
        axes[1].imshow(mask, cmap="Blues", vmin=0, vmax=1); axes[1].set_title("EMS water mask (ground truth)", fontsize=9)
        axes[2].imshow(pred, cmap="Blues", vmin=0, vmax=1); axes[2].set_title("U-Net prediction", fontsize=9)
        ov = rgb.copy()
        red = np.zeros_like(rgb); red[..., 0] = 1
        ov = np.where(pred[..., None] > 0.5, 0.55 * red + 0.45 * ov, ov)
        axes[3].imshow(ov); axes[3].set_title("Overlay (prediction)", fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.savefig(f"{vis_dir}/{name}.png", dpi=90)
        plt.close(fig)

    # training curve
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train loss", color="#4285FF")
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val loss", color="#25D9F2")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in history], [h["val_iou"] for h in history], label="val IoU", color="#22C55E")
    ax2.set_ylabel("val IoU")
    ax2.legend(loc="center right")
    plt.title("U-Net training (WorldFloods subset)")
    plt.savefig(f"{OUT}/unet_training_curve.png", dpi=110)
    plt.close(fig)

    report = {
        "model_name": "unet_water_segmentation",
        "version": "1.0.0",
        "architecture": "U-Net (base 16, BatchNorm, 5 input bands B,G,R,NIR,SWIR1)",
        "params": int(n_params),
        "epochs": EPOCHS, "patch": PATCH, "batch": BATCH, "lr": LR,
        "loss": "weighted BCE + Dice",
        "test_patch_metrics": {"iou": te[1], "dice": te[2], "precision": te[3],
                               "recall": te[4], "f1": te[5]},
        "scene_metrics": scene_metrics,
        "history": history,
        "data_provenance": {
            "source": "isp-uv-es/WorldFloodsv2 (HuggingFace, CC-BY-NC-4.0)",
            "paper": "Global flood extent segmentation in optical satellite images, Scientific Reports 13, 20316 (2023), DOI 10.1038/s41598-023-47595-7",
            "scenes": "15 train / 4 val / 4 test (different Copernicus EMS activations, event-level split)",
            "mask_definition": "Copernicus EMS flood/water polygons rasterized to S2 grid (water incl. permanent water)",
            "note": "Subset of full dataset (475 train scenes) due to CPU-only compute",
        },
        "artifact_path": ckpt,
    }
    with open(f"{OUT}/unet_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nsaved {ckpt}, {OUT}/unet_report.json, visuals in {vis_dir}")


if __name__ == "__main__":
    main()
