"""Plotting helpers.

Two rules that keep these images honest:

1. All images are shown with the SAME stretch, computed once from the ground
   truth. If every panel were stretched independently, a blurry prediction could
   look better than it is.
2. NIR is never shown inside an RGB picture. It is a separate grayscale panel.
   Pretending NIR is a colour channel produces pretty but meaningless images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

BAND_NAMES = ["Red", "Green", "Blue", "NIR"]


def stretch(img, low=2, high=98, ref=None):
    """Scale reflectance to 0..1 for display using percentile clipping.

    `ref` is the image the percentiles are taken from. Pass the ground truth so
    every panel in a comparison uses one identical stretch.
    """
    ref = img if ref is None else ref
    lo, hi = np.percentile(ref, low), np.percentile(ref, high)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def to_rgb(img, ref=None):
    """(H, W, 4) canonical [R, G, B, NIR] -> displayable (H, W, 3) RGB."""
    rgb = img[..., :3]
    ref_rgb = None if ref is None else ref[..., :3]
    return stretch(rgb, ref=ref_rgb)


def to_nir(img, ref=None):
    """(H, W, 4) -> displayable single-band NIR image."""
    ref_nir = None if ref is None else ref[..., 3]
    return stretch(img[..., 3], ref=ref_nir)


def show_pair(lr, hr, name="", save_path=None):
    """Side-by-side LR and HR, in RGB and in NIR."""
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(to_rgb(lr));            ax[0, 0].set_title("LR RGB {}".format(lr.shape[:2]))
    ax[0, 1].imshow(to_rgb(hr));            ax[0, 1].set_title("HR RGB {}".format(hr.shape[:2]))
    ax[1, 0].imshow(to_nir(lr), cmap="gray");  ax[1, 0].set_title("LR NIR")
    ax[1, 1].imshow(to_nir(hr), cmap="gray");  ax[1, 1].set_title("HR NIR")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def comparison_panel(lr, bicubic, pred, hr, name="", metrics=None, save_path=None):
    """The main figure: LR upsampled | Bicubic | FlexScaleSR | Ground truth.

    Top row is RGB, bottom row is NIR. All panels share the ground-truth stretch.
    Pass `pred=None` to show only the baseline comparison.
    """
    import cv2
    H, W = hr.shape[:2]
    lr_up = cv2.resize(lr, (W, H), interpolation=cv2.INTER_NEAREST)

    panels = [("LR (nearest)", lr_up), ("Bicubic", bicubic)]
    if pred is not None:
        panels.append(("FlexScaleSR", pred))
    panels.append(("Ground truth HR", hr))

    fig, ax = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 7))
    for j, (title, img) in enumerate(panels):
        ax[0, j].imshow(to_rgb(img, ref=hr))
        sub = ""
        if metrics and title in metrics:
            sub = "\nPSNR {:.2f}  SSIM {:.3f}".format(
                metrics[title]["psnr"], metrics[title]["ssim"])
        ax[0, j].set_title(title + " RGB" + sub, fontsize=10)
        ax[1, j].imshow(to_nir(img, ref=hr), cmap="gray")
        ax[1, j].set_title(title + " NIR", fontsize=10)
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_history(history, save_path=None):
    """Training curves: loss, PSNR, SSIM."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["epoch"], history["val_psnr"], color="tab:green")
    ax[1].set_title("validation PSNR (dB)"); ax[1].set_xlabel("epoch")
    ax[2].plot(history["epoch"], history["val_ssim"], color="tab:purple")
    ax[2].set_title("validation SSIM"); ax[2].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def error_map(pred, hr, name="", save_path=None):
    """Absolute error per band - shows WHERE the model is wrong."""
    err = np.abs(pred - hr)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    vmax = float(np.percentile(err, 99))
    for c in range(4):
        im = ax[c].imshow(err[..., c], cmap="magma", vmin=0, vmax=vmax)
        ax[c].set_title("|error| " + BAND_NAMES[c])
        ax[c].axis("off")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(name)
    _save(fig, save_path)
    return fig


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
