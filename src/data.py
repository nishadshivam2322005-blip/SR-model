"""Data loading for the satellite super-resolution dataset.

Two things matter here, and both were verified on the real files:

1. BAND ORDER
   LR tiles are stored as [B, G, R, NIR]   (Sentinel-2 B02, B03, B04, B08)
   HR tiles are stored as [R, G, B, NIR]
   Everything is converted to one canonical order: [R, G, B, NIR].
   So LR bands are permuted with [2, 1, 0, 3]. NIR (index 3) never moves.

2. RADIOMETRY
   LR is already float32 reflectance (roughly 0..1).
   HR is uint16 holding reflectance * 10000.
   HR is divided by `hr_scale` so both live in the same reflectance space.
   We never min-max normalise per image: that would destroy the real
   radiometric relationship between the two images.
"""

import os
import glob
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# LR is [B, G, R, NIR]; canonical order is [R, G, B, NIR].
LR_TO_CANONICAL = [2, 1, 0, 3]
CANONICAL_BAND_NAMES = ["Red", "Green", "Blue", "NIR"]


# --------------------------------------------------------------------------
# basic IO helpers
# --------------------------------------------------------------------------
def read_tif(path):
    """Read a TIFF as a numpy array shaped (H, W, C). Never modifies the file."""
    arr = tifffile.imread(path)
    if arr.ndim == 2:                      # single band -> add a channel axis
        arr = arr[:, :, None]
    return arr


def reorder_lr_bands(x):
    """LR [B, G, R, NIR] -> canonical [R, G, B, NIR].

    Works for numpy (H, W, C) and for torch (C, H, W).
    """
    if isinstance(x, torch.Tensor):
        return x[LR_TO_CANONICAL]          # channel-first
    return x[..., LR_TO_CANONICAL]         # channel-last


def lr_to_reflectance(arr):
    """LR is already reflectance; just make sure it is float32."""
    return arr.astype(np.float32)


def hr_to_reflectance(arr, hr_scale=10000.0):
    """HR is uint16 = reflectance * hr_scale. Divide to get reflectance."""
    return arr.astype(np.float32) / float(hr_scale)


def load_pair(lr_path, hr_path, hr_scale=10000.0):
    """Load one LR/HR pair, band-reordered and in reflectance units.

    Returns two numpy arrays shaped (H, W, 4) in canonical order [R, G, B, NIR].
    """
    lr = reorder_lr_bands(lr_to_reflectance(read_tif(lr_path)))
    hr = hr_to_reflectance(read_tif(hr_path), hr_scale)
    return lr, hr


def list_pairs(root, split):
    """Return [(name, lr_path, hr_path), ...] for files present in BOTH folders."""
    lr_dir = os.path.join(root, split, "LR")
    hr_dir = os.path.join(root, split, "HR")
    lr_names = {os.path.basename(p) for p in glob.glob(os.path.join(lr_dir, "*.tif"))}
    hr_names = {os.path.basename(p) for p in glob.glob(os.path.join(hr_dir, "*.tif"))}

    def sort_key(n):
        stem = os.path.splitext(n)[0]
        return (0, int(stem)) if stem.isdigit() else (1, 0)

    names = sorted(lr_names & hr_names, key=sort_key)
    return [(n, os.path.join(lr_dir, n), os.path.join(hr_dir, n)) for n in names]


# --------------------------------------------------------------------------
# augmentation: the SAME geometric transform is applied to LR and HR
# --------------------------------------------------------------------------
def paired_augment(lr, hr, rng):
    """Random horizontal flip, vertical flip and rotation.

    Both arrays are (H, W, C). Identical transforms keep LR and HR aligned.

    One detail worth knowing: a 90 or 270 degree rotation swaps height and
    width. Most HR tiles are not square (322 x 318, for example), so rotating
    them by 90 degrees would change their size and they could no longer be
    batched with the other tiles of that size. So quarter-turns are used only
    for square tiles; rectangular tiles get 0 or 180 degrees, which keeps the
    size unchanged. Combined with the two flips this still gives 8 variants for
    square tiles and 4 for rectangular ones.
    """
    if rng.random() < 0.5:                      # horizontal flip
        lr, hr = lr[:, ::-1], hr[:, ::-1]
    if rng.random() < 0.5:                      # vertical flip
        lr, hr = lr[::-1], hr[::-1]

    square = hr.shape[0] == hr.shape[1]
    k = int(rng.integers(0, 4)) if square else 2 * int(rng.integers(0, 2))
    if k:
        lr, hr = np.rot90(lr, k, (0, 1)), np.rot90(hr, k, (0, 1))
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SatelliteSRDataset(Dataset):
    """Pairs of (low-resolution input, high-resolution target).

    Every item is a dict:
        lr   : float tensor (4, 48, 48)   canonical [R, G, B, NIR], reflectance
        hr   : float tensor (4, H, W)     canonical [R, G, B, NIR], reflectance
        name : the file name, e.g. "1.tif"

    H and W come from the HR file and are NOT fixed - they range 307..336.
    """

    def __init__(self, root, split, hr_scale=10000.0, augment=False,
                 max_samples=None, seed=0):
        self.pairs = list_pairs(root, split)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        self.hr_scale = hr_scale
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def hr_shape(self, i):
        """(H, W) of the HR target - read from the header only, no pixel decoding."""
        with tifffile.TiffFile(self.pairs[i][2]) as tf:
            return tuple(tf.pages[0].shape[:2])

    def __getitem__(self, i):
        name, lr_path, hr_path = self.pairs[i]
        lr, hr = load_pair(lr_path, hr_path, self.hr_scale)
        if self.augment:
            lr, hr = paired_augment(lr, hr, self.rng)
        # (H, W, C) -> (C, H, W)
        lr = torch.from_numpy(lr.transpose(2, 0, 1).copy())
        hr = torch.from_numpy(hr.transpose(2, 0, 1).copy())
        return {"lr": lr, "hr": hr, "name": name}


# --------------------------------------------------------------------------
# Batching images whose HR size differs
# --------------------------------------------------------------------------
class SameSizeBatchSampler(Sampler):
    """Group images that share the same HR size into the same batch.

    HR tiles are 307..336 pixels wide, so arbitrary images cannot be stacked
    into one tensor. This sampler buckets the dataset by HR shape and only
    batches images from the same bucket. It is simple, and it keeps every
    target at its true size (the ground truth is never resized).
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        buckets = {}
        for i in range(len(dataset)):
            buckets.setdefault(dataset.hr_shape(i), []).append(i)
        self.buckets = list(buckets.values())

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            idx = list(bucket)
            if self.shuffle:
                self.rng.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                batches.append(idx[s:s + self.batch_size])
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum((len(b) + self.batch_size - 1) // self.batch_size
                   for b in self.buckets)


def collate(items):
    """Stack a list of items into one batch (all HR sizes are equal by design)."""
    return {
        "lr": torch.stack([d["lr"] for d in items]),
        "hr": torch.stack([d["hr"] for d in items]),
        "name": [d["name"] for d in items],
    }


def make_loader(dataset, batch_size=4, shuffle=True, seed=0, num_workers=0):
    """DataLoader that respects the variable HR size."""
    sampler = SameSizeBatchSampler(dataset, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler,
                      collate_fn=collate, num_workers=num_workers)


# ==========================================================================
# Dataset inspection
# --------------------------------------------------------------------------
# These functions never assume the dataset is correct. They read the real files
# and report what is actually there. Notebook 01 is a thin wrapper around them.
# ==========================================================================
def geo_info(path):
    """Pixel size, top-left corner, EPSG code and compression of a GeoTIFF."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value if "ModelPixelScaleTag" in tags else None
        tie = tags["ModelTiepointTag"].value if "ModelTiepointTag" in tags else None
        epsg = None
        if "GeoKeyDirectoryTag" in tags:
            keys = tags["GeoKeyDirectoryTag"].value
            # geo keys come as flat groups of 4; key 2048 is the geographic CRS
            for i in range(4, len(keys), 4):
                if keys[i] == 2048:
                    epsg = keys[i + 3]
        return {
            "pixel_size_x": None if scale is None else scale[0],
            "pixel_size_y": None if scale is None else scale[1],
            "origin_x": None if tie is None else tie[3],
            "origin_y": None if tie is None else tie[4],
            "epsg": epsg,
            "compression": str(page.compression),
            "nodata": tags["GDAL_NODATA"].value if "GDAL_NODATA" in tags else None,
        }


def inspect_file(path):
    """Shape, dtype, statistics and geo information for one TIFF."""
    arr = read_tif(path)
    a = arr.astype(np.float64)
    info = {
        "file": os.path.basename(path),
        "height": arr.shape[0], "width": arr.shape[1], "channels": arr.shape[2],
        "dtype": str(arr.dtype),
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "std": float(a.std()),
        "n_unique": int(np.unique(arr).size),
        "zero_fraction": float((arr == 0).mean()),
        "finite": bool(np.isfinite(a).all()),
    }
    info.update(geo_info(path))
    return info


def inspect_split(root, split, sub, limit=None):
    """inspect_file() over every TIFF in one folder. Returns a list of dicts."""
    paths = sorted(glob.glob(os.path.join(root, split, sub, "*.tif")))
    if limit:
        paths = paths[:limit]
    rows = []
    for p in paths:
        try:
            rows.append(inspect_file(p))
        except Exception as e:                      # a corrupted file lands here
            rows.append({"file": os.path.basename(p), "error": str(e)})
    return rows


def check_dataset(root, splits=("train", "val", "test")):
    """Structural health check. Returns a dict of findings, and prints nothing.

    Looks for missing pairs, duplicate names, wrong dtypes, wrong channel
    counts, corrupted files, and geographic overlap between splits (leakage).
    """
    report = {"splits": {}, "problems": [], "leakage": {}}
    boxes = {}

    for split in splits:
        lr_paths = glob.glob(os.path.join(root, split, "LR", "*.tif"))
        hr_paths = glob.glob(os.path.join(root, split, "HR", "*.tif"))
        lr_names = [os.path.basename(p) for p in lr_paths]
        hr_names = [os.path.basename(p) for p in hr_paths]
        lr_set, hr_set = set(lr_names), set(hr_names)

        if len(lr_names) != len(lr_set):
            report["problems"].append(split + "/LR has duplicate file names")
        if len(hr_names) != len(hr_set):
            report["problems"].append(split + "/HR has duplicate file names")

        missing_hr = sorted(lr_set - hr_set)
        missing_lr = sorted(hr_set - lr_set)
        for n in missing_hr:
            report["problems"].append(split + ": " + n + " has LR but no HR")
        for n in missing_lr:
            report["problems"].append(split + ": " + n + " has HR but no LR")

        lr_shapes, hr_shapes, lr_dtypes, hr_dtypes, corrupt = set(), set(), set(), set(), []
        split_boxes = []
        for name in sorted(lr_set & hr_set):
            lp = os.path.join(root, split, "LR", name)
            hp = os.path.join(root, split, "HR", name)
            try:
                lr, hr = read_tif(lp), read_tif(hp)
            except Exception as e:
                corrupt.append((name, str(e)))
                continue
            lr_shapes.add(lr.shape); hr_shapes.add(hr.shape)
            lr_dtypes.add(str(lr.dtype)); hr_dtypes.add(str(hr.dtype))
            if lr.shape[2] != 4 or hr.shape[2] != 4:
                report["problems"].append(split + "/" + name + " does not have 4 bands")
            g = geo_info(hp)
            if g["pixel_size_x"]:
                split_boxes.append((name,
                                    g["origin_x"], g["origin_y"],
                                    g["origin_x"] + hr.shape[1] * g["pixel_size_x"],
                                    g["origin_y"] - hr.shape[0] * g["pixel_size_y"]))

        for name, err in corrupt:
            report["problems"].append(split + "/" + name + " is unreadable: " + err)

        boxes[split] = split_boxes
        report["splits"][split] = {
            "n_lr": len(lr_set), "n_hr": len(hr_set), "n_paired": len(lr_set & hr_set),
            "lr_shapes": sorted(lr_shapes), "lr_dtypes": sorted(lr_dtypes),
            "hr_dtypes": sorted(hr_dtypes),
            "n_distinct_hr_shapes": len(hr_shapes),
            "hr_height_range": (min(s[0] for s in hr_shapes), max(s[0] for s in hr_shapes))
            if hr_shapes else None,
            "hr_width_range": (min(s[1] for s in hr_shapes), max(s[1] for s in hr_shapes))
            if hr_shapes else None,
            "corrupt": corrupt,
        }

    # geographic leakage: do footprints from two different splits overlap?
    def overlaps(a, b):
        w = min(a[3], b[3]) - max(a[1], b[1])
        h = min(a[2], b[2]) - max(a[4], b[4])
        return w > 0 and h > 0

    names = list(boxes)
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            hits = [(a[0], b[0]) for a in boxes[A] for b in boxes[B] if overlaps(a, b)]
            report["leakage"][A + " vs " + B] = len(hits)
            if hits:
                report["problems"].append("geographic overlap between " + A + " and " + B)

    report["ok"] = len(report["problems"]) == 0
    return report
