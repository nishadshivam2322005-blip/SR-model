"""Prediction and evaluation.

Contains the two "methods" we compare:

    bicubic_upsample()  - the non-neural baseline
    predict()           - FlexScaleSR

Both take the SAME inputs (band-reordered LR in reflectance) and produce output
at the SAME exact target size taken from the HR file. If they did not, the
comparison between them would be meaningless.

It can also be run from the command line:

    python -m src.inference --config configs/baseline.yaml --split test \\
        --checkpoint checkpoints/best.pt --save-tif
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import tifffile
import torch

from .data import list_pairs, load_pair, read_tif
from .model import build_model
from .metrics import compute_all, compute_per_band, summarise
from .train_utils import load_config, get_device, load_checkpoint


# --------------------------------------------------------------------------
# method 1: bicubic baseline
# --------------------------------------------------------------------------
def bicubic_upsample(lr, target_hw):
    """Resize an (h, w, 4) reflectance array to (H, W, 4) with bicubic interpolation."""
    H, W = int(target_hw[0]), int(target_hw[1])
    return cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# method 1b: bicubic + global sensor calibration
# --------------------------------------------------------------------------
# The LR and HR tiles come from different satellites. After dividing HR by
# 10000, HR is systematically darker than LR, and by a different amount in each
# band. That is a fixed property of the two sensors, not something a
# super-resolution model should have to discover.
#
# So we measure one number per band on the TRAINING SPLIT ONLY:
#
#     HR_band  =  gain_band  x  LR_band
#
# and apply those four numbers unchanged to validation and test. This separates
# "the model corrected the brightness" from "the model recovered real detail",
# which is the only way to tell whether super-resolution is actually working.
# --------------------------------------------------------------------------
def fit_calibration(root, split="train", hr_scale=10000.0):
    """Least-squares per-band gain mapping bicubic-upsampled LR onto HR.

    Fitted on `split` only (train by default), so applying it to val or test
    leaks nothing.
    """
    num = np.zeros(4)
    den = np.zeros(4)
    for name, lp, hp in list_pairs(root, split):
        lr, hr = load_pair(lp, hp, hr_scale)
        bic = bicubic_upsample(lr, hr.shape[:2]).reshape(-1, 4).astype(np.float64)
        ref = hr.reshape(-1, 4).astype(np.float64)
        num += (bic * ref).sum(axis=0)
        den += (bic * bic).sum(axis=0)
    return (num / den).astype(np.float32)


def apply_calibration(img, gains):
    """Multiply each band by its gain. img is (H, W, 4)."""
    return img * np.asarray(gains, dtype=np.float32)


def calibrated_bicubic(lr, target_hw, gains):
    """Baseline 1b: band reorder (already done) -> calibration -> bicubic."""
    return apply_calibration(bicubic_upsample(lr, target_hw), gains)


# --------------------------------------------------------------------------
# method 2: the trained network
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, lr, target_hw, device):
    """Run FlexScaleSR on one (h, w, 4) reflectance array -> (H, W, 4)."""
    model.eval()
    x = torch.from_numpy(lr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    y = model(x, (int(target_hw[0]), int(target_hw[1])))
    return y[0].float().cpu().numpy().transpose(1, 2, 0)


def load_model(checkpoint_path, config=None, device=None):
    """Build FlexScaleSR and load weights. Uses the config stored in the checkpoint."""
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = config or ckpt.get("config")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


# --------------------------------------------------------------------------
# saving predictions as GeoTIFF
# --------------------------------------------------------------------------
def _geo_tags(reference_tif):
    """Copy the geo tags of a reference TIFF so predictions stay georeferenced."""
    tags = []
    try:
        with tifffile.TiffFile(reference_tif) as tf:
            t = tf.pages[0].tags
            for name, code, dtype in [("ModelPixelScaleTag", 33550, "d"),
                                      ("ModelTiepointTag", 33922, "d"),
                                      ("GeoKeyDirectoryTag", 34735, "H"),
                                      ("GeoDoubleParamsTag", 34736, "d")]:
                if name in t:
                    v = t[name].value
                    tags.append((code, dtype, len(v), tuple(v), True))
            if "GeoAsciiParamsTag" in t:
                tags.append((34737, "s", 0, t["GeoAsciiParamsTag"].value, True))
    except Exception:
        pass
    return tags


def save_prediction(path, image, reference_tif=None):
    """Write a (H, W, 4) reflectance prediction as a float32 GeoTIFF.

    Bands are saved in canonical order [R, G, B, NIR] as reflectance, and the
    geotransform of the reference file is copied when one is given.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(
        path,
        image.astype(np.float32),
        photometric="minisblack",
        planarconfig="contig",
        compression="deflate",
        extratags=_geo_tags(reference_tif) if reference_tif else (),
        description=("FlexScaleSR prediction; bands = Red, Green, Blue, NIR; "
                     "units = surface reflectance"),
    )


# --------------------------------------------------------------------------
# evaluation over a whole split
# --------------------------------------------------------------------------
def evaluate_split(root, split, hr_scale=10000.0, model=None, device=None,
                   save_tif_dir=None, max_samples=None, gains=None):
    """Score the baselines and (optionally) FlexScaleSR on one split.

    Pass `gains` (from fit_calibration) to also score the calibrated bicubic
    baseline.

    Returns
    -------
    per_image : DataFrame, one row per image per method
    per_band  : DataFrame, one row per image per method per band
    """
    pairs = list_pairs(root, split)
    if max_samples:
        pairs = pairs[:max_samples]

    rows, band_rows = [], []
    for name, lr_path, hr_path in pairs:
        lr, hr = load_pair(lr_path, hr_path, hr_scale)
        target_hw = hr.shape[:2]

        outputs = {"Bicubic": bicubic_upsample(lr, target_hw)}
        if gains is not None:
            outputs["Bicubic+calib"] = calibrated_bicubic(lr, target_hw, gains)
        if model is not None:
            outputs["FlexScaleSR"] = predict(model, lr, target_hw, device)

        for method, pred in outputs.items():
            m = compute_all(pred, hr, lr_shape=lr.shape[:2])
            m.update({"name": name, "method": method,
                      "height": target_hw[0], "width": target_hw[1]})
            rows.append(m)
            for b in compute_per_band(pred, hr):
                b.update({"name": name, "method": method})
                band_rows.append(b)

        if save_tif_dir and "FlexScaleSR" in outputs:
            save_prediction(os.path.join(save_tif_dir, name),
                            outputs["FlexScaleSR"], reference_tif=hr_path)

    return pd.DataFrame(rows), pd.DataFrame(band_rows)


def summary_table(per_image):
    """Mean of every metric for each method - the table that goes in the report."""
    cols = ["psnr", "ssim", "mae", "rmse", "sam", "ergas"]
    return per_image.groupby("method")[cols].mean().round(4)


# --------------------------------------------------------------------------
# command line entry point
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate / run FlexScaleSR")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save-tif", action="store_true",
                    help="write predicted GeoTIFFs to <out>/predictions/<split>")
    ap.add_argument("--bicubic-only", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="also score the calibrated bicubic baseline "
                         "(gains fitted on the train split only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"].get("device", "auto"))
    root = cfg["data"]["root"]
    hr_scale = cfg["data"]["hr_scale"]

    model = None
    if not args.bicubic_only:
        if not os.path.exists(args.checkpoint):
            raise SystemExit("checkpoint not found: " + args.checkpoint
                             + " (train first, or pass --bicubic-only)")
        model, cfg, _ = load_model(args.checkpoint, device=device)

    gains = None
    if args.calibrate:
        gains = fit_calibration(root, "train", hr_scale)
        print("per-band gains fitted on train: " +
              ", ".join("%s %.4f" % (b, g) for b, g in
                        zip(["R", "G", "B", "NIR"], gains)))

    tif_dir = os.path.join(args.out, "predictions", args.split) if args.save_tif else None
    per_image, per_band = evaluate_split(root, args.split, hr_scale,
                                         model, device, save_tif_dir=tif_dir,
                                         gains=gains)

    os.makedirs(args.out, exist_ok=True)
    per_image.to_csv(os.path.join(args.out, "metrics_" + args.split + ".csv"), index=False)
    per_band.to_csv(os.path.join(args.out, "metrics_per_band_" + args.split + ".csv"),
                    index=False)

    table = summary_table(per_image)
    print("\n" + args.split + " results (" + str(per_image["name"].nunique()) + " images)")
    print(table.to_string())

    with open(os.path.join(args.out, "summary_" + args.split + ".json"),
              "w", encoding="utf-8") as f:
        json.dump({m: summarise(g.to_dict("records"))
                   for m, g in per_image.groupby("method")}, f, indent=2)


if __name__ == "__main__":
    main()
