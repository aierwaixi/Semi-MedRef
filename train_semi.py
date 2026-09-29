#!/usr/bin/env python3
"""Official training entry point for the Semi-MedRef release.

The implementation intentionally exposes only the two primary segmentation
architectures reported in the paper: MMI-UNet and GuideDecoder.  Configuration
files contain the paper defaults; command-line overrides are provided for the
label ratio, random seed, device, and output directory.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as vision_transforms
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandZoomd,
    Resized,
    ToTensord,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler

from engine.wrapper_semi import (
    LanGuideMedSeg_SemiWrapper,
    MMIUNet_SemiWrapper,
)
from utils import config as config_utils
from utils.dataset import MosMed, QaTa
from utils.dataset_semi import (
    MosMedSemiLabeled,
    MosMedSemiUnlabeled,
    QaTaSemiLabeled,
    QaTaSemiUnlabeled,
)


SUPPORTED_RATIOS = ("0.01", "0.02", "0.05", "0.15", "1.0")
WRAPPERS = {
    "mmiunet": MMIUNet_SemiWrapper,
    "guidedecoder": LanGuideMedSeg_SemiWrapper,
}


def parse_cli() -> argparse.Namespace:
    """Parse reproducible training overrides without modifying the YAML files."""
    parser = argparse.ArgumentParser(description="Train Semi-MedRef")
    parser.add_argument("--config", required=True, help="YAML experiment file")
    parser.add_argument("--ratio", choices=SUPPORTED_RATIOS, default=None)
    parser.add_argument(
        "--model-arch",
        choices=tuple(WRAPPERS),
        default=None,
        help="Override MODEL.model_arch from the YAML file",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=int, default=None, help="Visible CUDA index")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Lightning checkpoint to resume")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two train/validation batches to verify the installation",
    )
    return parser.parse_args()


def _ratio_from_path(path: str) -> str:
    match = re.search(r"labeled_([0-9]*\.?[0-9]+)", os.path.basename(path))
    return match.group(1) if match else "custom"


def _apply_ratio(cfg, ratio: str) -> None:
    """Select the released split manifests without changing validation/test."""
    dataset = str(cfg.dataset).lower()
    if dataset == "qata":
        cfg.train_csv_path = f"./data_splits/QaTa/split_labeled_{ratio}.json"
        cfg.unlabeled_csv_path = f"./data_splits/QaTa/split_unlabeled_{ratio}.json"
    elif dataset == "mosmed":
        if ratio == "1.0":
            full = "./data_splits/MosMed/Train_text_MosMedData+ 1(in).plabel.csv"
            cfg.train_csv_path = full
            cfg.unlabeled_csv_path = full
        else:
            complement = {"0.01": "0.99", "0.02": "0.98", "0.05": "0.95", "0.15": "0.85"}[ratio]
            cfg.train_csv_path = f"./data_splits/MosMed/labeled_{ratio}.plabel.csv"
            cfg.unlabeled_csv_path = f"./data_splits/MosMed/unlabeled_{complement}.plabel.csv"
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset!r}")

    if ratio == "1.0":
        # Preserve the paired-loader interface while disabling every objective
        # that depends on an unlabeled pool.
        cfg.unsup_weight = 0.0
        cfg.itc_w_unsup = 0.0
        cfg.use_xpatchmix = False


def load_config(cli: argparse.Namespace):
    """Load one released recipe and apply explicit command-line overrides."""
    cfg = config_utils.load_cfg_from_cfg_file(cli.config)
    if cli.ratio is not None:
        _apply_ratio(cfg, cli.ratio)
    if cli.model_arch is not None:
        cfg.model_arch = cli.model_arch
    if cli.seed is not None:
        cfg.seed = cli.seed
    if cli.device is not None:
        cfg.device = [cli.device]
    if cli.output_dir is not None:
        cfg.model_save_path = cli.output_dir
    if cli.num_workers is not None:
        cfg.num_workers = cli.num_workers

    cfg.model_arch = str(cfg.model_arch).lower()
    aliases = {"medseg": "guidedecoder", "languide": "guidedecoder"}
    cfg.model_arch = aliases.get(cfg.model_arch, cfg.model_arch)
    if cfg.model_arch not in WRAPPERS:
        raise ValueError(
            f"Unsupported model_arch={cfg.model_arch!r}; choose {sorted(WRAPPERS)}"
        )

    # The training field is a short run label, never a checkpoint path.
    cfg.model_save_filename = cfg.model_arch
    if not hasattr(cfg, "num_workers"):
        cfg.num_workers = 8
    return cfg


def build_weak_transform(image_size):
    """Weak view used by the EMA teacher (paper implementation details)."""
    return Compose(
        [
            LoadImaged(["image", "gt"], reader="PILReader"),
            EnsureChannelFirstd(["image", "gt"]),
            RandZoomd(
                ["image", "gt"],
                min_zoom=0.95,
                max_zoom=1.20,
                mode=["bicubic", "nearest"],
                prob=0.10,
            ),
            Resized(["image"], spatial_size=image_size, mode="bicubic"),
            Resized(["gt"], spatial_size=image_size, mode="nearest"),
            NormalizeIntensityd(["image"], channel_wise=True),
            ToTensord(["image", "gt", "token", "mask"]),
        ]
    )


class StrongImageAugmentation(torch.nn.Module):
    """Photometric strong view used by the student."""

    def __init__(self, color_probability=0.8, blur_probability=0.5):
        super().__init__()
        self.color_probability = float(color_probability)
        self.blur_probability = float(blur_probability)
        self.color = vision_transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.blur = vision_transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))

    def forward(self, image):
        """Apply stochastic color jitter and blur to the student's image view."""
        if torch.rand(()) < self.color_probability:
            image = self.color(image)
        if torch.rand(()) < self.blur_probability:
            image = self.blur(image)
        return image


def build_datasets(cfg):
    """Build labeled, unlabeled, and fixed-validation datasets for one run."""
    weak = build_weak_transform(cfg.image_size)
    strong = StrongImageAugmentation()
    common = dict(
        root_path=cfg.train_root_path,
        tokenizer=cfg.bert_type,
        mode="train",
        image_size=cfg.image_size,
    )

    if cfg.dataset == "qata":
        labeled = QaTaSemiLabeled(
            csv_path=cfg.train_csv_path, labeled_tf=weak, **common
        )
        unlabeled = QaTaSemiUnlabeled(
            csv_path=cfg.unlabeled_csv_path,
            weak_tf=weak,
            strong_aug=strong,
            use_pos_aug=cfg.use_pos_aug,
            pos_aug_mode=cfg.pos_aug_mode,
            pos_aug_p=cfg.pos_aug_p,
            use_pos_aug_geosync=cfg.use_pos_aug_geosync,
            **common,
        )
        validation = QaTa(
            csv_path=cfg.valid_csv_path,
            root_path=cfg.valid_root_path,
            tokenizer=cfg.bert_type,
            mode="valid",
            image_size=cfg.image_size,
        )
    elif cfg.dataset == "mosmed":
        labeled = MosMedSemiLabeled(
            csv_path=cfg.train_csv_path, labeled_tf=weak, **common
        )
        unlabeled = MosMedSemiUnlabeled(
            csv_path=cfg.unlabeled_csv_path,
            weak_tf=weak,
            strong_aug=strong,
            use_pos_aug=cfg.use_pos_aug,
            pos_aug_mode=cfg.pos_aug_mode,
            pos_aug_p=cfg.pos_aug_p,
            **common,
        )
        validation = MosMed(
            csv_path=cfg.valid_csv_path,
            root_path=cfg.valid_root_path,
            tokenizer=cfg.bert_type,
            mode="valid",
            image_size=cfg.image_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset!r}")
    return labeled, unlabeled, validation


def build_loaders(cfg, ratio: str):
    """Create deterministic paired training loaders and a validation loader."""
    labeled, unlabeled, validation = build_datasets(cfg)
    workers = int(cfg.num_workers)
    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    if ratio == "1.0":
        labeled_loader = DataLoader(
            labeled,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    else:
        unlabeled_steps = math.ceil(len(unlabeled) / cfg.train_batch_size)
        sampler = RandomSampler(
            labeled,
            replacement=True,
            num_samples=unlabeled_steps * cfg.train_batch_size,
        )
        labeled_loader = DataLoader(
            labeled,
            batch_size=cfg.train_batch_size,
            sampler=sampler,
            drop_last=True,
            **loader_kwargs,
        )

    unlabeled_loader = DataLoader(
        unlabeled,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    combined = CombinedLoader(
        {"labeled": labeled_loader, "unlabeled": unlabeled_loader},
        mode=getattr(cfg, "combined_mode", "max_size_cycle"),
    )
    return combined, validation_loader


def module_kwargs(cfg):
    """Arguments shared by the core and optional ISPG training modules."""
    return dict(
        bert_type=cfg.bert_type,
        vision_type=cfg.vision_type,
        project_dim=cfg.project_dim,
        lr=cfg.lr,
        ema_decay=cfg.ema_decay_semi,
        burn_in_epochs=cfg.burn_in_epochs,
        unsup_weight=cfg.unsup_weight,
        conf_th=cfg.conf_th,
        load_convnext_ckpt=getattr(cfg, "convnext_checkpoint", ""),
        unsup_rampup_epochs=cfg.unsup_rampup_epochs,
        ema_decay_start=cfg.ema_decay_start,
        ema_decay_end=cfg.ema_decay_end,
        ema_warmup_epochs=cfg.ema_warmup_epochs,
        enable_itc=cfg.enable_itc,
        itc_weight=cfg.itc_weight,
        itc_tau=cfg.itc_tau,
        itc_w_unsup=cfg.itc_w_unsup,
        use_xpatchmix=cfg.use_xpatchmix,
        mix_block=cfg.mix_block,
        mix_prob=cfg.mix_prob,
        xpatchmix_mode=cfg.xpatchmix_mode,
        mix_margin=cfg.mix_margin,
        viz_every=getattr(cfg, "visualization_interval", 0),
        viz_dir=getattr(cfg, "visualization_dir", "./outputs/visualizations"),
        pseudo_threshold_mode=cfg.pseudo_threshold_mode,
        pseudo_threshold_temp=cfg.pseudo_threshold_temp,
    )


def build_module(cfg):
    """Instantiate the MMI-UNet or GuideDecoder Semi-MedRef wrapper."""
    return WRAPPERS[cfg.model_arch](**module_kwargs(cfg))


def run_training(cfg, ratio: str, cli, module, train_loader, validation_loader):
    """Configure logging/checkpointing and execute a Lightning run."""
    if cli.dry_run:
        module.burn_in_epochs = 0
        module.unsup_rampup_epochs = 0

    run_name = f"semimedref-{cfg.model_arch}-{cfg.dataset}-r{ratio}-s{cfg.seed}"
    output_root = Path(cfg.model_save_path)
    checkpoint_dir = output_root / "checkpoints" / run_name
    log_root = output_root / "logs"

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename=run_name + "-{epoch:03d}-{val_loss:.4f}-{val_dice:.4f}",
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=int(cfg.patience),
        min_delta=1e-3,
    )
    callbacks = [
        checkpoint,
        early_stopping,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    loggers = [
        TensorBoardLogger(save_dir=log_root, name=run_name),
        CSVLogger(save_dir=log_root, name=run_name + "-csv"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.device,
        min_epochs=int(cfg.min_epochs),
        max_epochs=int(cfg.max_epochs),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        deterministic=True,
        accumulate_grad_batches=int(getattr(cfg, "accumulate_grad_batches", 1)),
        precision=getattr(cfg, "precision", 32),
        fast_dev_run=2 if cli.dry_run else False,
    )
    trainer.fit(module, train_loader, validation_loader, ckpt_path=cli.resume)


def main() -> None:
    """Configure and launch one Semi-MedRef training run."""
    cli = parse_cli()
    cfg = load_config(cli)
    ratio = cli.ratio or _ratio_from_path(cfg.train_csv_path)

    pl.seed_everything(int(cfg.seed), workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, validation_loader = build_loaders(cfg, ratio)
    module = build_module(cfg)
    run_training(cfg, ratio, cli, module, train_loader, validation_loader)


if __name__ == "__main__":
    main()
