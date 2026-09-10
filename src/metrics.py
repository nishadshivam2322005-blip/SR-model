"""Super-resolution quality metrics.

Everything here works on numpy arrays shaped (H, W, C) in the SAME reflectance
representation for prediction and ground truth. That is the whole point: if one
side were reflectance and the other raw uint16, every number would be nonsense.

These are IMAGE QUALITY metrics, not segmentation metrics. IoU and Dice do not
apply to this problem and are deliberately absent.

    PSNR  (dB, higher better)  - overall pixel accuracy on a log scale
    SSIM  (0..1, higher)       - structural / perceptual similarity
    MAE   (lower)              - mean absolute error, in reflectance units
    RMSE  (lower)              - root mean squared error, penalises big misses
    SAM   (degrees, lower)     - spectral angle: is the colour/spectrum right?
    ERGAS (lower)              - a standard remote-sensing summary of relative
                                 error across all bands, scaled by the
                                 resolution ratio between LR and HR
"""

import numpy as np
from skimage.metrics import structural_similarity

# Nominal reflectance range used for PSNR and SSIM. Reflectance is normally
# 0..1; a few specular pixels exceed it, which is expected and harmless.
DATA_RANGE = 1.0


def _as_float(a):
    return np.asarray(a, dtype=np.float64)


def mae(pred, target):
    """Mean absolute error."""
    return float(np.abs(_as_float(pred) - _as_float(target)).mean())


def rmse(pred, target):
    """Root mean squared error."""
    return float(np.sqrt(((_as_float(pred) - _as_float(target)) ** 2).mean()))


def psnr(pred, target, data_range=DATA_RANGE):
    """Peak signal-to-noise ratio in decibels."""
    mse = ((_as_float(pred) - _as_float(target)) ** 2).mean()
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(pred, target, data_range=DATA_RANGE):
    """Mean structural similarity over the 4 bands."""
    pred, target = _as_float(pred), _as_float(target)
    return float(structural_similarity(
        target, pred, data_range=data_range, channel_axis=-1))


# A pixel whose 4-band vector is shorter than this is effectively black. The
# spectral ANGLE of a zero vector does not exist, so such pixels are excluded
# from SAM rather than being given a meaningless value.
SAM_MIN_NORM = 1e-3


def sam(pred, target, min_norm=SAM_MIN_NORM, return_coverage=False):
    """Spectral Angle Mapper, averaged over valid pixels, in DEGREES.

    Why the mask matters. SAM is the angle between the 4-band vector of the
    prediction and the 4-band vector of the truth. If either vector is (near)
    zero it has no direction, and the naive formula `dot / (norm + eps)` then
    returns cos = 0, i.e. a confident 90 degrees, for pixels that may in fact be
    identical. Dark water and shadow are common in this dataset, so that silently
    inflates the score. Pixels shorter than `min_norm` are skipped instead.

    Set return_coverage=True to also get the fraction of pixels actually used.
    """
    pred, target = _as_float(pred), _as_float(target)
    n_pred = np.linalg.norm(pred, axis=-1)
    n_true = np.linalg.norm(target, axis=-1)
    valid = (n_pred > min_norm) & (n_true > min_norm)

    if not valid.any():
        return (0.0, 0.0) if return_coverage else 0.0

    dot = (pred * target).sum(axis=-1)[valid]
    cos = np.clip(dot / (n_pred[valid] * n_true[valid]), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos)).mean())
    if return_coverage:
        return angle, float(valid.mean())
    return angle


def ergas(pred, target, ratio, min_mean=1e-3):
    """ERGAS - relative dimensionless global error.

    ERGAS = 100 * (h/l) * sqrt( mean_over_bands( (RMSE_b / mean_b)^2 ) )

    `ratio` is h/l, the HR pixel size divided by the LR pixel size, which for
    this dataset is lr_height / hr_height, about 48/320 = 0.15.

    KNOWN WEAKNESS, kept because this is the standard definition: each band's
    error is divided by that band's MEAN. A scene whose Blue band averages 0.01
    reflectance (deep water, dark shadow) produces a huge ERGAS even when the
    absolute error is small. On this dataset two validation scenes do exactly
    that and drag the mean from about 9 up to about 37. Always read the MEDIAN
    ERGAS alongside the mean, and use `dark_band_images()` to find the scenes
    responsible. `min_mean` only prevents a division by zero; it does not
    rescue an image whose band mean is genuinely tiny.
    """
    pred, target = _as_float(pred), _as_float(target)
    per_band = []
    for c in range(target.shape[-1]):
        m = abs(target[..., c].mean())
        e = np.sqrt(((pred[..., c] - target[..., c]) ** 2).mean())
        per_band.append((e / max(m, min_mean)) ** 2)
    return float(100.0 * ratio * np.sqrt(np.mean(per_band)))


def dark_band_images(records, threshold=0.02):
    """Names of images whose darkest band mean is below `threshold`.

    These are the scenes whose ERGAS should not be trusted. `records` must carry
    a "min_band_mean" field, which compute_all() adds when given the target.
    """
    return [r["name"] for r in records
            if r.get("min_band_mean", 1.0) < threshold]


def compute_all(pred, target, lr_shape=None, data_range=DATA_RANGE):
    """All six metrics for one image pair, returned as a dict.

    pred, target : (H, W, C) arrays in the same reflectance representation
    lr_shape     : (h, w) of the LR input, used for the ERGAS ratio.
                   If omitted the nominal 1/6.67 ratio is used.
    """
    ratio = (lr_shape[0] / target.shape[0]) if lr_shape else (1.0 / 6.67)
    angle, coverage = sam(pred, target, return_coverage=True)
    return {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "sam": angle,
        "ergas": ergas(pred, target, ratio),
        # diagnostics: how much of the image SAM could use, and how dark the
        # darkest band is (an ERGAS warning sign)
        "sam_coverage": coverage,
        "min_band_mean": float(_as_float(target).reshape(-1, target.shape[-1]).mean(0).min()),
    }


def compute_per_band(pred, target, band_names=("Red", "Green", "Blue", "NIR"),
                     data_range=DATA_RANGE):
    """PSNR / SSIM / MAE / RMSE for each band separately.

    SAM and ERGAS are not included because both are defined ACROSS bands and
    are meaningless for a single band on its own.
    """
    rows = []
    for c, name in enumerate(band_names):
        p, t = pred[..., c], target[..., c]
        rows.append({
            "band": name,
            "psnr": psnr(p, t, data_range),
            "ssim": float(structural_similarity(
                _as_float(t), _as_float(p), data_range=data_range)),
            "mae": mae(p, t),
            "rmse": rmse(p, t),
        })
    return rows


def summarise(records, keys=("psnr", "ssim", "mae", "rmse", "sam", "ergas")):
    """mean / median / std for a list of per-image metric dicts."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in records], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "std": float(v.std())}
    return out
