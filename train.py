"""
train.py — trains the CMAS audio-visual deepfake detector (or an ablation
via --mode visual_only / audio_only).

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --mode visual_only --epochs 10

All real metrics (accuracy, precision, recall, F1, ROC-AUC) are computed on
the validation split after every epoch and logged to logs/train.log and
TensorBoard. No numbers in this file are hardcoded or fabricated — they only
exist once you actually run this script against real data.
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import FakeAVCelebDataset, collate_fn, make_balanced_sampler
from models.fusion import CMASDeepfakeDetector
from utils.logging_utils import get_logger
from utils.seed import get_device, set_seed

# torch.cuda.amp.{autocast,GradScaler} are deprecated in favor of the
# device-agnostic torch.amp namespace (torch>=2.3). Fall back to the old
# namespace on older torch so this still runs on torch 2.1/2.2 (our
# requirements.txt floor), rather than crashing with an ImportError on a
# supported version.
try:
    from torch.amp import GradScaler, autocast

    def _make_scaler(enabled: bool):
        return GradScaler("cuda", enabled=enabled)
except ImportError:  # pragma: no cover - only exercised on torch < 2.3
    from torch.cuda.amp import GradScaler as _LegacyGradScaler
    from torch.cuda.amp import autocast as _legacy_autocast

    def autocast(device_type: str, dtype=None, enabled: bool = True):  # type: ignore
        return _legacy_autocast(enabled=enabled, dtype=dtype)

    def _make_scaler(enabled: bool):
        return _LegacyGradScaler(enabled=enabled)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CMAS deepfake detector.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["multimodal", "visual_only", "audio_only"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--backup_dir", type=str, default=None,
        help="If set, copy the checkpoint here (via shutil.copy2) every time a new best is "
             "saved during training — e.g. a mounted Google Drive path. On Colab, a session "
             "disconnect wipes /content/ entirely; without a live backup, a checkpoint that was "
             "successfully saved locally minutes before a disconnect can still be lost along with "
             "everything else in /content/. This does not slow down training meaningfully since "
             "checkpoints only save on improvement, not every epoch. Example: "
             "--backup_dir /content/drive/MyDrive/CMAS-Project/outputs/checkpoints",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    return cfg


def build_model(cfg: dict, mode: str, device: torch.device) -> CMASDeepfakeDetector:
    m = cfg["model"]
    model = CMASDeepfakeDetector(
        visual_backbone=m["visual_backbone"],
        visual_pretrained=m["visual_pretrained"],
        audio_model_name=m["audio_model_name"],
        fusion_type=m["fusion_type"],
        fusion_hidden_dim=m["fusion_hidden_dim"],
        fusion_num_heads=m["fusion_num_heads"],
        fusion_dropout=m["fusion_dropout"],
        num_classes=m["num_classes"],
        mode=mode,
        freeze_backbones=False,
    )
    return model.to(device)


def prepare_batch(batch: dict, model: CMASDeepfakeDetector, device: torch.device, sample_rate: int = 16000):
    """Runs modality-specific preprocessing and moves tensors to device.

    `sample_rate` must match config.yaml's `audio.sample_rate` (the rate the
    waveform was extracted/cached at in dataset.py) — it is a parameter here,
    not hardcoded, specifically so callers propagate the configured value
    instead of silently assuming 16kHz if config.yaml is ever changed."""
    visual_frames_input = None
    audio_input = None

    if model.mode != "audio_only":
        frames = batch["frames"].to(device)  # (B,T,H,W,3) uint8
        visual_frames_input = model.visual_encoder.preprocess(frames)

    if model.mode != "visual_only":
        waveform = batch["waveform"].numpy()  # (B, num_samples)
        audio_tensor = model.audio_encoder.preprocess(waveform, sample_rate=sample_rate).to(device)
        audio_input = audio_tensor

    labels = batch["label"].to(device)
    return visual_frames_input, audio_input, labels


@torch.no_grad()
def evaluate(model: CMASDeepfakeDetector, loader: DataLoader, device: torch.device, criterion: nn.Module, sample_rate: int = 16000):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:  # entire mini-batch was unreadable files — see dataset.py collate_fn
            continue
        visual_input, audio_input, labels = prepare_batch(batch, model, device, sample_rate=sample_rate)
        logits = model(visual_input, audio_input)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(FAKE)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = {
        # Explicit float() casts: scikit-learn's return type for these
        # scalar metrics is not consistent across versions — some versions
        # return plain Python float, others return numpy.float64. The
        # latter breaks torch.load() on PyTorch >=2.6, whose default
        # `weights_only=True` refuses to unpickle numpy scalar types for
        # security reasons (reproduced and confirmed while debugging this:
        # a checkpoint saved with a numpy.float64 hiding in val_metrics
        # fails to load with "Unsupported global: numpy._core.multiarray.
        # scalar"). Casting here means the checkpoint only ever contains
        # plain Python types, regardless of which sklearn version produced
        # them, and is loadable even under the strict weights_only=True
        # default without needing to bypass it.
        "loss": float(total_loss / max(n_batches, 1)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
    }
    # ROC-AUC requires both classes present in the eval split
    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def compute_class_weights(dataset: FakeAVCelebDataset, device: torch.device) -> torch.Tensor:
    labels = dataset.df["label"].map({"REAL": 0, "FAKE": 1}).values
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = counts.sum() / (2.0 * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    mode = args.mode or "multimodal"
    run_name = args.run_name or f"{mode}_{int(time.time())}"

    # early_stopping_metric drives `if current_metric > best_metric` below,
    # i.e. it assumes higher-is-better. "loss" is lower-is-better, so
    # silently accepting cfg["training"]["early_stopping_metric"] == "val_loss"
    # would track the *worst* epoch, not the best. Fail fast instead.
    _HIGHER_IS_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    _monitor_key = cfg["training"]["early_stopping_metric"].replace("val_", "")
    if _monitor_key not in _HIGHER_IS_BETTER:
        raise ValueError(
            f"config.yaml training.early_stopping_metric='{cfg['training']['early_stopping_metric']}' "
            f"is not supported: must be one of {sorted('val_' + m for m in _HIGHER_IS_BETTER)} "
            f"(all higher-is-better; 'val_loss' is NOT supported since checkpointing logic below "
            f"assumes higher current_metric is better)."
        )

    # Real bug found in practice: writing scientific notation without a
    # decimal point in config.yaml (e.g. `learning_rate: 1e-4` instead of
    # `1.0e-4`) is valid YAML syntax, but PyYAML's float resolver requires
    # the decimal point to recognize it as a number — without one, it
    # silently parses as the STRING "1e-4" instead of the float 0.0001.
    # That doesn't fail at config-load time; it fails three call-frames deep
    # inside torch's LR scheduler with a confusing "can't multiply sequence
    # by non-int of type 'float'" TypeError, because a Python string is a
    # "sequence" type too. Catch it here instead, at startup, with a message
    # that actually points at the real problem.
    _NUMERIC_TRAINING_FIELDS = [
        "learning_rate", "backbone_learning_rate", "weight_decay", "warmup_ratio",
        "grad_clip_norm", "label_smoothing",
    ]
    for _field in _NUMERIC_TRAINING_FIELDS:
        _val = cfg["training"][_field]
        if not isinstance(_val, (int, float)):
            raise TypeError(
                f"config.yaml training.{_field}={_val!r} was parsed as {type(_val).__name__}, not a number. "
                f"If this looks like a number written in scientific notation without a decimal point "
                f"(e.g. '1e-4'), that's the bug: PyYAML requires a decimal point to parse it as a float "
                f"('1.0e-4', not '1e-4') — without one it silently becomes a string. Fix it in config.yaml."
            )

    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["auto_detect"])

    logger = get_logger("train", cfg["paths"]["log_dir"], filename=f"train_{run_name}.log")
    logger.info(f"Run: {run_name} | mode={mode} | device={device}")
    logger.info(f"Config: {cfg}")

    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- data
    train_ds = FakeAVCelebDataset(
        cfg["paths"]["train_manifest"],
        cache_dir=cfg["paths"]["cache_dir"],
        num_frames=cfg["model"]["num_frames"],
        sample_rate=cfg["audio"]["sample_rate"],
        max_duration_sec=cfg["audio"]["max_duration_sec"],
        split="train",
        augment_cfg=cfg["augmentation"],
    )
    val_ds = FakeAVCelebDataset(
        cfg["paths"]["val_manifest"],
        cache_dir=cfg["paths"]["cache_dir"],
        num_frames=cfg["model"]["num_frames"],
        sample_rate=cfg["audio"]["sample_rate"],
        max_duration_sec=cfg["audio"]["max_duration_sec"],
        split="val",
    )

    sampler = make_balanced_sampler(train_ds) if cfg["sampling"]["balanced"] else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------ model
    model = build_model(cfg, mode, device)

    if cfg["training"]["class_weighted_loss"]:
        class_weights = compute_class_weights(train_ds, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["training"]["label_smoothing"])
        logger.info(f"Using class-weighted loss with weights={class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])

    # Discriminative learning rates: lower LR for pretrained backbones, higher for new heads
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "visual_encoder.backbone" in name or "audio_encoder.model" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["training"]["backbone_learning_rate"]},
            {"params": head_params, "lr": cfg["training"]["learning_rate"]},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_enabled = cfg["training"]["mixed_precision"] and device.type == "cuda"
    amp_dtype_str = cfg.get("device", {}).get("mixed_precision_dtype", "float16")
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype_str, torch.float16)
    # GradScaler's loss-scaling is only needed for float16 (bfloat16 has enough
    # dynamic range that scaling is unnecessary and can be actively harmful);
    # this reads config.device.mixed_precision_dtype, which was previously an
    # unused/dead config field — wiring it up here so it actually does what
    # the config file implies instead of always silently using float16.
    scaler = _make_scaler(enabled=amp_enabled and amp_dtype == torch.float16)

    writer = None
    if cfg["logging"]["use_tensorboard"] and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(cfg["paths"]["log_dir"], "tb", run_name))

    # Freeze backbones for the first N epochs (gradual unfreezing), then unfreeze.
    freeze_epochs = cfg["training"]["freeze_backbone_epochs"]

    best_metric = -np.inf
    best_state = None
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        # Order matters: model.train() recursively sets every submodule
        # (including a frozen backbone) back into train() mode, which would
        # silently re-enable BatchNorm running-stat updates on a "frozen"
        # backbone if called after set_backbone_trainable(). Call it first,
        # then let set_backbone_trainable() override the backbone back to
        # eval() when appropriate. (This ordering bug was caught during
        # review — an earlier version called these two lines in the reverse
        # order.)
        model.train()
        model.set_backbone_trainable(epoch >= freeze_epochs)
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{run_name}] epoch {epoch+1}/{cfg['training']['epochs']}")
        for batch in pbar:
            if batch is None:  # entire mini-batch was unreadable files — see dataset.py collate_fn
                continue
            visual_input, audio_input, labels = prepare_batch(batch, model, device, sample_rate=cfg["audio"]["sample_rate"])

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(visual_input, audio_input)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

            if writer and global_step % cfg["logging"]["log_every_n_steps"] == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, criterion, sample_rate=cfg["audio"]["sample_rate"])

        logger.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        monitor_key = cfg["training"]["early_stopping_metric"].replace("val_", "")
        current_metric = val_metrics[monitor_key]

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"best_model_{mode}.pt")
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": cfg,
                    "mode": mode,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            logger.info(f"New best {monitor_key}={best_metric:.4f}. Saved checkpoint to {ckpt_path}")

            if args.backup_dir:
                # See parse_args()'s --backup_dir help: a Colab disconnect
                # wipes /content/ entirely, so a checkpoint saved locally
                # seconds before a disconnect can still be lost. Copying
                # immediately after every improvement (not just at the end
                # of training) means at most one epoch's worth of progress
                # is ever at risk, not the whole run. Wrapped in try/except
                # since a transient Drive I/O hiccup shouldn't kill training.
                try:
                    os.makedirs(args.backup_dir, exist_ok=True)
                    backup_path = os.path.join(args.backup_dir, f"best_model_{mode}.pt")
                    shutil.copy2(ckpt_path, backup_path)
                    logger.info(f"Backed up checkpoint to {backup_path}")
                except OSError as e:
                    logger.warning(f"Checkpoint backup to {args.backup_dir} failed ({e}); continuing training anyway.")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= cfg["training"]["early_stopping_patience"]:
            logger.info(f"Early stopping triggered after epoch {epoch+1}.")
            break

    if writer:
        writer.close()

    logger.info(f"Training complete. Best {cfg['training']['early_stopping_metric']}={best_metric:.4f}")
    return os.path.join(cfg["paths"]["checkpoint_dir"], f"best_model_{mode}.pt")


if __name__ == "__main__":
    main()
