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

from tayr.errors import ConfigError


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
    tiles and 0.2 overlap that is ~42 forward passes per frame - see docs/RESEARCH.md
    section 3.2 before raising `overlap`.
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
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    input_size: Annotated[int, Field(gt=0)] = 640
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25

    @model_validator(mode="after")
    def _checkpoint_needs_checksum(self) -> DetectorConfig:
        if self.checkpoint is not None and self.checkpoint_sha256 is None:
            raise ValueError(
                "checkpoint_sha256 is required whenever a checkpoint is set. Weights are "
                "loaded with weights_only=True, but a recorded checksum is what proves the "
                "file is the one the manifest claims."
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
    epochs: Annotated[int, Field(gt=0)] = 50
    batch_size: Annotated[int, Field(gt=0)] = 8
    learning_rate: Annotated[float, Field(gt=0)] = 1e-4
    resume_from: Path | None = None


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

    @model_validator(mode="after")
    def _grouping_is_mandatory(self) -> EvalConfig:
        if not self.group_splits_by_video:
            raise ValueError(
                "group_splits_by_video=False lets frames from one track appear in both "
                "train and test, which inflates every reported number. If you genuinely "
                "need ungrouped splits, do it in a throwaway script, not a run config."
            )
        return self


class Config(_Strict):
    """Top-level run config."""

    seed: Annotated[int, Field(ge=0, lt=2**32)] = 1337
    output_dir: Path = Path("runs")
    datasets: list[DatasetConfig] = Field(default_factory=list)
    region_proposal: RegionProposalConfig = RegionProposalConfig()
    slicing: SliceConfig = SliceConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

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
