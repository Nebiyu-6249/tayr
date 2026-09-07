"""`tayr train`: seeded, resumable, manifested RF-DETR training.

## What this module is responsible for

RF-DETR owns the optimisation. This module owns the parts that decide whether a number
coming out of it can be believed:

  * the **seed**, applied before anything imports a dataloader
  * the **device**, resolved once and recorded, refusing to quietly run on CPU when the
    config asked for a GPU (see `tayr.devices`)
  * the **manifest**, written *before* training starts - a crash at epoch 40 still
    leaves behind the commit, seed and config that produced the checkpoints on disk
  * the **name mapping** from Tayr's config keys onto RF-DETR's, in exactly one place

## The mapping is the fragile part

`rfdetr_train_kwargs` is a pure function with no torch import, so it can be tested on
any machine. `test_train_detector.py` checks every key it emits against the *installed*
`rfdetr.config.TrainConfig` model fields, which means an upstream rename fails the suite
here rather than silently reverting a hyperparameter to its default mid-run. RF-DETR
sets `extra="forbid"` on that model specifically so a typo raises
[VERIFIED: rfdetr 1.9.4, rfdetr/config.py:1019-1023], and this test is the same idea
one layer out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tayr.config import Config
from tayr.determinism import seed_everything
from tayr.devices import ResolvedDevice, resolve_device
from tayr.errors import ConfigError, DependencyUnavailableError
from tayr.manifest import RunManifest, build_manifest

#: Kwargs `RFDETR.train()` pops before building its `TrainConfig`, so they are valid to
#: pass but are not fields on that model.
#: [VERIFIED: rfdetr 1.9.4, rfdetr/detr.py:283-286 pops `device`, :290 pops `resolution`]
ABSORBED_KWARGS: frozenset[str] = frozenset({"device", "resolution"})

#: RF-DETR's dataset loader for the `train/`, `valid/`, `test/` +
#: `_annotations.coco.json` layout that `tayr dataset convert` produces.
#: [VERIFIED: rfdetr 1.9.4, rfdetr/datasets/coco.py:1254-1268]
DATASET_FILE = "roboflow"

#: Checkpoint filenames RF-DETR writes that omit optimizer and scheduler state.
#: Resuming from one restarts the optimizer cold.
#: [VERIFIED: rfdetr 1.9.4, rfdetr/detr.py:919-925]
LIGHTWEIGHT_CHECKPOINTS: frozenset[str] = frozenset(
    {
        "checkpoint_best_regular.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_total.pth",
        "last_ema.pth",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Where a training run put things, and what it was given."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    device: ResolvedDevice
    rfdetr_kwargs: dict[str, Any]
    variant: str
    resumed_from: Path | None = None
    checkpoints: list[Path] = field(default_factory=list)


def new_run_id(variant: str, *, now: datetime | None = None) -> str:
    """Sortable run id. Time first so `ls` on the runs directory is chronological."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-rfdetr-{variant}"


def rfdetr_train_kwargs(cfg: Config, *, device: str, output_dir: Path) -> dict[str, Any]:
    """Map a Tayr `Config` onto the kwargs `RFDETR.train()` accepts.

    Pure: no torch, no filesystem, no network. `device` comes in already resolved so
    this function never has to decide what `auto` means.
    """
    train = cfg.train
    if train.dataset_dir is None:
        raise ConfigError(
            "train.dataset_dir is not set. Training needs a directory containing "
            "train/, valid/ and test/ subdirectories, each with images and an "
            "_annotations.coco.json. Produce one with `tayr dataset convert`."
        )

    kwargs: dict[str, Any] = {
        "dataset_dir": str(train.dataset_dir),
        "dataset_file": DATASET_FILE,
        "output_dir": str(output_dir),
        "device": device,
        "epochs": train.epochs,
        "batch_size": train.batch_size,
        "grad_accum_steps": train.grad_accum_steps,
        "lr": train.learning_rate,
        "lr_encoder": train.lr_encoder,
        "weight_decay": train.weight_decay,
        "warmup_epochs": train.warmup_epochs,
        "num_workers": train.num_workers,
        "checkpoint_interval": train.checkpoint_interval,
        "eval_interval": train.eval_interval,
        "early_stopping": train.early_stopping,
        "early_stopping_patience": train.early_stopping_patience,
        "use_ema": train.use_ema,
        "tensorboard": train.tensorboard,
        # Set explicitly rather than inherited. These three send run data to external
        # services; leaving them to an upstream default means a future release of
        # rfdetr could start doing that without anything in this repo changing.
        "wandb": False,
        "mlflow": False,
        "clearml": False,
    }
    if train.resolution is not None:
        kwargs["resolution"] = train.resolution
    if train.resume_from is not None:
        kwargs["resume"] = str(train.resume_from)
    return kwargs


def _resume_note(resume_from: Path | None) -> list[str]:
    """Warn in the manifest when a resume will not restore optimizer state."""
    if resume_from is None:
        return []
    if resume_from.name in LIGHTWEIGHT_CHECKPOINTS:
        return [
            f"RESUMED FROM {resume_from.name}, one of RF-DETR's lightweight checkpoints. "
            "Those omit optimizer and LR-scheduler state to stay small, so both restart "
            "cold and the loss curve of this run is not continuous with the previous "
            "one. Resume from last.ckpt or checkpoint_<epoch>.ckpt to avoid that."
        ]
    return [f"RESUMED FROM {resume_from}."]


def train_detector(
    cfg: Config,
    *,
    config_path: Path | None = None,
    run_id: str | None = None,
    repo: Path | None = None,
    synthetic: bool = False,
    dry_run: bool = False,
) -> TrainingRun:
    """Train a detector from a config.

    Writes the manifest before training starts, so a run that dies mid-epoch still
    leaves behind what produced the checkpoints next to it.

    `dry_run` does everything except call RF-DETR: it seeds, resolves the device, builds
    the kwargs and writes the manifest. It exists so the whole wiring can be exercised
    on a machine with no GPU and no dataset, and so a config can be checked against a
    real device before a rented GPU starts charging.
    """
    if cfg.detector.backend != "rfdetr":
        raise ConfigError(
            f"detector.backend={cfg.detector.backend!r} has no training implementation. "
            "Only 'rfdetr' is implemented; see docs/RESEARCH.md section 14 for why."
        )

    seed_report = seed_everything(cfg.seed, strict_torch=True)
    device = resolve_device(cfg.device)

    variant = cfg.detector.variant
    resolved_run_id = run_id or new_run_id(variant)
    run_dir = cfg.output_dir / resolved_run_id
    resume_from = cfg.train.resume_from

    kwargs = rfdetr_train_kwargs(cfg, device=device.device, output_dir=run_dir)

    notes = [
        f"device: {device.note}",
        f"rfdetr variant: {variant}",
        *_resume_note(resume_from),
    ]
    if device.fell_back:
        notes.append(
            "DEVICE FELL BACK TO CPU. Timing and throughput here describe CPU execution "
            "and are not comparable with a GPU run of the same config."
        )
    if dry_run:
        notes.append("DRY RUN. No training was performed; no checkpoint here is trained.")

    manifest = build_manifest(
        run_id=resolved_run_id,
        command="tayr train",
        config=cfg.to_dict(),
        seed_report=seed_report,
        config_path=config_path,
        repo=repo,
        synthetic=synthetic,
        notes=notes,
    )
    manifest_path = _write_manifest(manifest, run_dir, resuming=resume_from is not None)

    if not dry_run:
        model = _build_model(cfg, device=device.device)
        model.train(**kwargs)

    return TrainingRun(
        run_id=resolved_run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        device=device,
        rfdetr_kwargs=kwargs,
        variant=variant,
        resumed_from=resume_from,
        checkpoints=sorted(run_dir.glob("*.pth")) + sorted(run_dir.glob("*.ckpt")),
    )


def _write_manifest(manifest: RunManifest, run_dir: Path, *, resuming: bool) -> Path:
    """Write `manifest.json`, or a numbered resume manifest when one already exists."""
    if not resuming or not (run_dir / "manifest.json").exists():
        return manifest.write(run_dir)
    existing = len(list(run_dir.glob("manifest.resume-*.json")))
    return manifest.write(run_dir, filename=f"manifest.resume-{existing + 1:03d}.json")


def _build_model(cfg: Config, *, device: str) -> Any:
    """Construct the RF-DETR variant that training will run against."""
    from tayr.detection.rfdetr_backend import _variant_class, verify_checkpoint

    detector = cfg.detector
    build_kwargs: dict[str, Any] = {"device": device}

    if detector.pretrain_weights is not None:
        # DetectorConfig already refuses a weight path with no checksum, so this branch
        # should be unreachable. It is a raise rather than an assert because asserts
        # vanish under `python -O`, and losing this one would mean calling the verifier
        # with no expected digest - which is to say, not verifying anything.
        if detector.pretrain_weights_sha256 is None:
            raise ConfigError(
                "detector.pretrain_weights is set with no pretrain_weights_sha256. "
                "DetectorConfig is supposed to make this unreachable; reaching it means "
                "the config was constructed without validation."
            )
        verify_checkpoint(detector.pretrain_weights, detector.pretrain_weights_sha256)
        build_kwargs["pretrain_weights"] = str(detector.pretrain_weights)

    try:
        return _variant_class(detector.variant)(**build_kwargs)
    except ImportError as exc:
        # RF-DETR raises ImportError from train() when its `train` extra is missing.
        raise DependencyUnavailableError(
            f"RF-DETR could not be constructed: {exc}. Training needs the RF-DETR "
            "training stack (pytorch-lightning and friends); install the 'cv' extra."
        ) from exc
