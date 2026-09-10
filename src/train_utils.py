"""Config loading, seeding, device choice, train/validate loops and checkpoints.

Nothing clever here on purpose - it is a plain PyTorch training loop that a
first-year student can read from top to bottom.
"""

import os
import time
import random
import numpy as np
import torch
import yaml

from .metrics import psnr, ssim


# --------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------
def load_config(path):
    """Read a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer="auto"):
    """Pick CUDA when it is available, otherwise CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy_image(t):
    """Torch (C, H, W) -> numpy (H, W, C) for the metric functions."""
    return t.detach().float().cpu().numpy().transpose(1, 2, 0)


# --------------------------------------------------------------------------
# one epoch of training
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler=None):
    """Run one pass over the training set. Returns mean loss and mean parts."""
    model.train()
    total, parts_sum, n = 0.0, {}, 0

    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        target_hw = (hr.shape[-2], hr.shape[-1])   # exact size comes from the HR image

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:                     # mixed precision, GPU only
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(lr, target_hw)
            loss, parts = loss_fn(pred.float(), hr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr, target_hw)
            loss, parts = loss_fn(pred, hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v
        n += 1

    n = max(n, 1)
    return total / n, {k: v / n for k, v in parts_sum.items()}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Returns mean validation loss, mean PSNR and mean SSIM."""
    model.eval()
    total, psnrs, ssims, n = 0.0, [], [], 0

    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred = model(lr, (hr.shape[-2], hr.shape[-1]))
        loss, _ = loss_fn(pred, hr)
        total += loss.item()
        n += 1
        for i in range(pred.shape[0]):
            p = to_numpy_image(pred[i])
            t = to_numpy_image(hr[i])
            psnrs.append(psnr(p, t))
            ssims.append(ssim(p, t))

    return (total / max(n, 1),
            float(np.mean(psnrs)) if psnrs else 0.0,
            float(np.mean(ssims)) if ssims else 0.0)


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config,
                    run=None):
    """Save everything needed to resume training later.

    `run` records the provenance of this checkpoint: which experiment produced
    it, how many epochs were planned, and how many images it was trained on.
    Without it a 3-epoch smoke run and a 100-epoch real run look identical on
    disk, and a trial checkpoint can quietly end up in a results table.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "run": run or {},
    }, path)


def describe_checkpoint(path):
    """One-line human summary of a checkpoint, plus whether it is a final model.

    Returns (text, is_final). `is_final` is True only when the checkpoint says
    it came from a run tagged "full" that trained on the whole training split.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    run = ck.get("run", {})
    kind = run.get("run_type", "unknown")
    planned = run.get("planned_epochs", "?")
    n_train = run.get("n_train", "?")
    is_final = (kind == "full" and isinstance(n_train, int) and n_train >= 100)
    text = ("{}  run_type={}  epoch {}/{}  trained on {} images  val PSNR {}"
            .format(os.path.basename(path), kind, ck.get("epoch", "?"), planned,
                    n_train, round(ck.get("metrics", {}).get("val_psnr", float("nan")), 3)))
    return text, is_final


def assert_final_checkpoint(path):
    """Raise unless `path` is a checkpoint from a real full training run.

    Call this before reporting results, so a trial checkpoint cannot be
    mistaken for a final model.
    """
    text, is_final = describe_checkpoint(path)
    if not is_final:
        raise RuntimeError(
            "REFUSING to treat this as a final model:\n  " + text +
            "\nThis checkpoint did not come from a full training run. Run "
            "section H of notebook 03, then re-run this notebook.")
    return text


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Restore a checkpoint. Returns the saved dict so you can read epoch/metrics."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# --------------------------------------------------------------------------
# the full training loop
# --------------------------------------------------------------------------
def fit(model, train_loader, val_loader, loss_fn, config, device,
        epochs=None, ckpt_dir="checkpoints", resume=None, verbose=True,
        run_type="trial"):
    """Train the model, keeping the best checkpoint by validation PSNR.

    `run_type` is stamped into every checkpoint. Use "full" only for a real
    training run on the whole training split; anything else is treated as a
    trial and will be rejected by assert_final_checkpoint().

    Returns a history dict of lists that the notebook plots directly.
    """
    tcfg = config["training"]
    epochs = epochs or tcfg["epochs"]

    run_info = {
        "run_type": run_type,
        "planned_epochs": epochs,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "batch_size": tcfg["batch_size"],
        "learning_rate": tcfg["learning_rate"],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=tcfg["learning_rate"],
                                 weight_decay=tcfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = bool(tcfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch, best_psnr, bad_epochs = 1, -float("inf"), 0
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("metrics", {}).get("best_psnr", -float("inf"))
        if verbose:
            print("resumed from " + resume + " at epoch " + str(start_epoch))

    patience = tcfg.get("early_stopping_patience", 0)
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_psnr": [], "val_ssim": [], "lr": []}

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model, train_loader, loss_fn,
                                        optimizer, device, scaler)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose:
            print("epoch {:3d}/{}  train {:.5f}  val {:.5f}  "
                  "PSNR {:6.2f}  SSIM {:.4f}  ({:.1f}s)".format(
                      epoch, epochs, train_loss, val_loss,
                      val_psnr, val_ssim, time.time() - t0))

        metrics = {"val_loss": val_loss, "val_psnr": val_psnr,
                   "val_ssim": val_ssim, "best_psnr": max(best_psnr, val_psnr)}
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer,
                        scheduler, epoch, metrics, config, run_info)

        # Model selection uses VALIDATION only. Test data is never touched here.
        if val_psnr > best_psnr:
            best_psnr, bad_epochs = val_psnr, 0
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer,
                            scheduler, epoch, metrics, config, run_info)
            if verbose:
                print("          new best (PSNR {:.2f}) -> best.pt".format(best_psnr))
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                if verbose:
                    print("early stopping: no improvement for "
                          + str(patience) + " epochs")
                break

    return history
