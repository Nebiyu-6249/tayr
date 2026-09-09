"""Config schema and loader.

Every run is driven by a version-controlled YAML file. Nothing that changes a result
is passed as a bare CLI flag, and no path is hardcoded - the same config must run
unchanged on a laptop and on a rented cloud GPU.

Unknown keys are a hard error. A typo'd `learing_rate` that silently falls back to a
default is exactly the kind of bug that produces an inexplicable result three weeks
later, so `extra="forbid"` is set on every model here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tayr.devices import validate_device_spec
from tayr.errors import ConfigError

# RF-DETR variant classes this project will construct. Taken from the installed
# package's own export list, not from memory.
# [VERIFIED: rfdetr 1.9.4, rfdetr/variants.py __all__ and class definitions]
#
# RFDETRBase is deliberately absent: it carries @deprecated_class(deprecated_in="1.7.0",
# remove_in="2.0.0") in that same file, so building a dissertation on it would tie the
# project to a class scheduled for deletion.
RFDETR_VARIANTS = ("nano", "small", "medium", "large")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    """Where a dataset lives and what it is allowed to be used for.

    `redistributable` is not decoration. Drone-vs-Bird is distributed under a data
    usage agreement granting no redistribution rights, so anything derived from it
    must never reach the public repository. Converters check this flag.
    """

    name: str
    root: Path
    annotation_format: Literal["coco", "yolo", "dvb", "dut", "antiuav", "voc"]
    splits: dict[str, str] = Field(default_factory=dict)
    redistributable: bool = False
    licence: str = "UNKNOWN"
    citation: str | None = None


class SliceConfig(_Strict):
    """Sliced (tiled) inference geometry.

    Cost scales with tile count, which scales with 1/(1-overlap)^2. At 4K with 640px
    tiles and 0.2 overlap that is an 8x4 grid - 32 forward passes per frame, so ~960 per
    second at 30fps. See docs/RESEARCH.md section 3.2 before raising `overlap`, and
    `test_slicing.py::test_tile_count_matches_the_documented_cost_model` for the
    arithmetic.
    """

    enabled: bool = True
    tile_width: Annotated[int, Field(gt=0)] = 640
    tile_height: Annotated[int, Field(gt=0)] = 640
    overlap: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.2


class RegionProposalConfig(_Strict):
    """Motion-based region proposal, to avoid running the detector over empty sky.

    This stage has a recall ceiling: anything it misses, the detector never sees. Its
    own miss rate is measured and reported separately, or a 5% proposer miss would be
    invisible inside end-to-end mAP and look like a detector failure.
    """

    enabled: bool = False
    mode: Literal["static", "egomotion"] = "static"
    min_blob_area: Annotated[int, Field(gt=0)] = 4


class DetectorConfig(_Strict):
    """Detector backend.

    Default is RF-DETR (Apache-2.0). Ultralytics is deliberately absent: it is
    AGPL-3.0 and would relicense this entire project, including the network-served
    app. See docs/RESEARCH.md section 2.
    """

    backend: Literal["rfdetr", "dfine"] = "rfdetr"
    variant: Literal["nano", "small", "medium", "large"] = "nano"
    """Which RF-DETR size to build. Ignored by other backends.

    `nano` is the default because it is the only variant that trains at a tolerable
    speed on CPU, and a default that cannot be run on the machine in front of you is a
    default that never gets exercised."""

    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    pretrain_weights: Path | None = None
    """Starting weights for training. When None, RF-DETR downloads its own published
    checkpoint for the chosen variant over the network at first use."""

    pretrain_weights_sha256: str | None = None
    input_size: Annotated[int, Field(gt=0)] = 640
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25

    @model_validator(mode="after")
    def _checkpoints_need_checksums(self) -> DetectorConfig:
        """Any local weight file named here must carry a recorded checksum.

        Weights are loaded with `weights_only=True`, which stops a malicious pickle from
        executing, but it does not tell you the file is the one the manifest claims. The
        checksum does, and a result whose weights cannot be identified is not a result.
        """
        for path_field, sum_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("pretrain_weights", "pretrain_weights_sha256"),
        ):
            if getattr(self, path_field) is not None and getattr(self, sum_field) is None:
                raise ValueError(
                    f"{sum_field} is required whenever {path_field} is set. Weights are "
                    "loaded with weights_only=True, but a recorded checksum is what "
                    "proves the file is the one the manifest claims."
                )
        return self


class TrackerConfig(_Strict):
    """ByteTrack-style association plus a constant-velocity Kalman filter.

    `low_threshold` is the point of the whole design: detections between low and high
    are kept and matched against existing tracks rather than discarded, which is how a
    target that fades below threshold keeps its identity.
    """

    high_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    low_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    max_age: Annotated[int, Field(gt=0)] = 30
    min_hits: Annotated[int, Field(gt=0)] = 3
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    centre_distance_factor: Annotated[float, Field(ge=0.0)] = 2.0
    """Size-normalised centre-distance fallback for association, in box widths.

    IoU association has a hard displacement ceiling of about 54% of a box's side at
    IoU>=0.3, regardless of absolute size. For a 10px target that is 5.4px per frame -
    less than a bird flapping at 5Hz actually moves. Small fast targets therefore break
    tracks under pure IoU, which is exactly the regime this project studies.

    When IoU association fails, a detection whose centre lies within
    `centre_distance_factor` box widths of the predicted track position is still
    eligible. Set to 0 to disable and get pure ByteTrack IoU behaviour."""

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> TrackerConfig:
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be below high_threshold "
                f"({self.high_threshold}); otherwise the low-score association pass that "
                "ByteTrack exists for can never fire."
            )
        return self


class TrainConfig(_Strict):
    """Detector training.

    Every key here changes a result and therefore lives in the config rather than on the
    command line. The names are Tayr's; `tayr.train.detector` maps them onto the RF-DETR
    `TrainConfig` field names, and that mapping is the one place where a rename in the
    upstream package can break us.

    Deliberately NOT exposed, though RF-DETR accepts them: `batch_size="auto"` (its probe
    runs a forward+backward pass whose result depends on whatever else is on the GPU, so
    two runs of the same config could train at different effective batch sizes), and the
    wandb/mlflow/clearml loggers (network egress from a training run, to services this
    project does not use).
    """

    dataset_dir: Path | None = None
    """Root of a detector-ready dataset: `train/`, `valid/` and `test/` subdirectories,
    each holding images and an `_annotations.coco.json`.

    [VERIFIED: rfdetr 1.9.4, rfdetr/datasets/coco.py:1265-1268 - the "roboflow" dataset
    layout maps split -> (root/train, root/train/_annotations.coco.json) and maps `val`
    onto the directory named `valid`.] Build one with `tayr dataset prepare`."""

    epochs: Annotated[int, Field(gt=0)] = 50
    batch_size: Annotated[int, Field(gt=0)] = 8
    grad_accum_steps: Annotated[int, Field(gt=0)] = 4
    """Effective batch size is `batch_size * grad_accum_steps`. Raising this instead of
    `batch_size` is how a 16GB P100 trains at an effective batch it cannot hold."""

    learning_rate: Annotated[float, Field(gt=0)] = 1e-4
    lr_encoder: Annotated[float, Field(gt=0)] = 1.5e-4
    """Separate learning rate for the pretrained backbone."""

    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-4
    warmup_epochs: Annotated[float, Field(ge=0.0)] = 0.0
    resolution: Annotated[int, Field(gt=0)] | None = None
    """Square input resolution. None keeps the variant's own default.

    RF-DETR rejects any value not divisible by `patch_size * num_windows` for the chosen
    variant, and raises rather than rounding [VERIFIED: rfdetr/detr.py:302-311]."""

    num_workers: Annotated[int, Field(ge=0)] = 2
    checkpoint_interval: Annotated[int, Field(gt=0)] = 10
    eval_interval: Annotated[int, Field(gt=0)] = 1
    early_stopping: bool = False
    early_stopping_patience: Annotated[int, Field(gt=0)] = 10
    use_ema: bool = True
    tensorboard: bool = False
    """Off by default. RF-DETR defaults it on, but tensorboard is in its `loggers`
    extra, which this project does not pin - leaving the upstream default in place would
    make training depend on a package that may not be installed."""

    resume_from: Path | None = None
    """A checkpoint to continue from.

    Pass the trainer's own `last.ckpt` or `checkpoint_<epoch>.ckpt` to resume optimizer
    and scheduler state too. RF-DETR's four `checkpoint_best_*.pth` files deliberately
    omit that state to stay small, so resuming from one restarts the optimizer cold
    [VERIFIED: rfdetr/detr.py:919-928]."""


class EvalConfig(_Strict):
    """Evaluation protocol.

    `iou_thresholds_small` is separate deliberately. IoU@0.5 on an 8px box needs the
    prediction within about a pixel, which is inside annotation noise, so the small
    buckets are additionally reported at 0.25. Any comparison against a published
    number must state its protocol next to it - different splits are not comparable.
    """

    iou_thresholds: tuple[float, ...] = (0.5,)
    iou_thresholds_small: tuple[float, ...] = (0.25, 0.5)
    group_splits_by_video: bool = True

    coco_sweep: bool = True
    """Also report mAP averaged over IoU 0.50:0.05:0.95, the COCO primary metric.

    Averaging over ten thresholds costs ten matching passes and is worth it: AP@0.5
    alone rewards a detector that finds objects roughly, and this project's whole
    subject is objects small enough that "roughly" is most of the box."""

    negatives_dir: Path | None = None
    """Directory of frames from footage guaranteed to contain no target.

    Every detection here is a false positive by construction, so this needs no
    annotation. mAP without a false-alarm rate describes half the system."""

    negatives_fps: Annotated[float, Field(gt=0)] | None = None
    """Frame rate of the negatives, when they are loose frames.

    A per-hour rate is frames/fps/3600, so a wrong fps scales the headline number
    linearly and silently and is never guessed. It is **not** required when
    `negatives_dir` holds videos: the frame rate is read from the containers, which is
    both more accurate and impossible to get wrong. Which case applies cannot be known
    until the directory is read, so the check lives in the harness rather than here."""

    @model_validator(mode="after")
    def _grouping_is_mandatory(self) -> EvalConfig:
        if not self.group_splits_by_video:
            raise ValueError(
                "group_splits_by_video=False lets frames from one track appear in both "
                "train and test, which inflates every reported number. If you genuinely "
                "need ungrouped splits, do it in a throwaway script, not a run config."
            )
        return self

    @model_validator(mode="after")
    def _a_frame_rate_needs_footage(self) -> EvalConfig:
        if self.negatives_fps is not None and self.negatives_dir is None:
            raise ValueError("negatives_fps is set but negatives_dir is not.")
        return self


class Config(_Strict):
    """Top-level run config."""

    seed: Annotated[int, Field(ge=0, lt=2**32)] = 1337
    device: str = "auto"
    """`auto`, `cpu`, `cuda`, `cuda:<index>` or `mps`.

    `auto` may fall back to CPU and records that it did. An explicit accelerator that is
    not present raises - see `tayr.devices`. Validated for shape here so a typo is
    caught by `tayr config validate` on a machine with no torch installed."""

    output_dir: Path = Path("runs")
    datasets: list[DatasetConfig] = Field(default_factory=list)
    region_proposal: RegionProposalConfig = RegionProposalConfig()
    slicing: SliceConfig = SliceConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _device_is_recognised(self) -> Config:
        validate_device_spec(self.device)
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config. Raises ConfigError with the file path attached."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{p} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{p} must contain a YAML mapping at the top level, got {type(raw).__name__}"
        )
    try:
        return Config.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{p} failed validation:\n{exc}") from exc
