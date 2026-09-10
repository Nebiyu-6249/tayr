"""`tayr eval`: run a detector over a split and produce the numbers that matter.

## Why this is not RF-DETR's `evaluate()`

RF-DETR has its own evaluation, and it reports mAP@0.50, mAP@0.50:0.95 and a per-class
table [VERIFIED: rfdetr 1.9.4, rfdetr/detr.py:1038-1080]. Two of the four numbers this
project needs are not in it and cannot be:

  * **pixels-on-target buckets.** The research hypothesis is a claim about what happens
    below ~20px on target. A single overall mAP averages exactly the distinction the
    project exists to measure.
  * **false alarms per hour on drone-free footage.** No annotation exists for negatives
    and no COCO evaluator has a concept of them, yet this is the number that decides
    whether a system like this is deployable.

So the detector is run through Tayr's own `Detector` interface and scored by
`tayr.eval.detection`. The trade-off is stated in the report itself: this module's AP is
all-point interpolated, `pycocotools` uses 101-point, and the two are not directly
comparable.

## Threshold discipline

Detections are collected once, at the *lowest* confidence that any reported metric
needs, and the false-alarm count is thresholded afterwards. Running inference twice at
two thresholds would score two different detection sets and let a favourable pair be
chosen; collecting once means the mAP and the false-alarm rate always describe the same
model behaviour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from tayr.config import Config
from tayr.datasets.coco import CocoSplit, read_coco_split
from tayr.determinism import seed_everything
from tayr.devices import ResolvedDevice, resolve_device
from tayr.errors import ConfigError
from tayr.eval.detection import (
    COCO_SWEEP,
    ImagePrediction,
    evaluate_dataset,
    sweep_average_precision,
)
from tayr.eval.false_alarms import false_alarms_per_hour
from tayr.eval.report import EvaluationReport
from tayr.geometry import SizeBucket
from tayr.manifest import build_manifest
from tayr.worker.detector import Detection

#: Which split directory each split name lives in. RF-DETR maps `val` onto `valid/`
#: [VERIFIED: rfdetr 1.9.4, rfdetr/datasets/coco.py:1265-1268]; the same mapping is used
#: here so `tayr eval` and RF-DETR's own validation read the same files.
SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

#: Buckets that additionally get the low IoU threshold. At 8px, IoU@0.5 needs the box
#: within about a pixel, which is inside annotation noise.
SMALL_BUCKETS: frozenset[SizeBucket] = frozenset({SizeBucket.TINY, SizeBucket.SMALL})

#: Image suffixes the negatives loader will read.
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#: Video suffixes the negatives loader will decode. Decoding happens through
#: `tayr.worker.pipeline.iter_frames`, which probes the container before decoding a
#: single frame - the resource-exhaustion rule applies to owner-supplied media too, and
#: a mistyped path to a 16K file should fail at the probe rather than in the OOM killer.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


class SupportsDetect(Protocol):
    """The slice of `tayr.worker.detector.Detector` this harness uses."""

    @property
    def is_real(self) -> bool: ...
    @property
    def name(self) -> str: ...
    def detect(self, frame: npt.NDArray[np.uint8]) -> Detection: ...


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Everything one `tayr eval` produced."""

    report: EvaluationReport
    n_images: int
    n_detections: int
    seconds: float
    device: ResolvedDevice | None = None
    manifest_path: Path | None = None
    report_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def images_per_second(self) -> float:
        return self.n_images / self.seconds if self.seconds > 0 else 0.0


def load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    """Read an image as HxWx3 RGB uint8.

    OpenCV decodes to BGR. Handing that to RF-DETR, which documents RGB channel order
    [VERIFIED: rfdetr/detr.py:2224-2226], silently costs accuracy in a way that looks
    like a bad model rather than a bad conversion - so the swap happens here, once.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigError(f"could not decode image: {path}")
    return np.ascontiguousarray(image[:, :, ::-1])


def split_annotation_path(dataset_dir: Path, split: str) -> Path:
    """Locate a split's COCO file, raising with the valid names rather than guessing."""
    if split not in SPLIT_DIRS:
        raise ConfigError(
            f"unknown split {split!r}; expected one of {', '.join(sorted(SPLIT_DIRS))}"
        )
    return dataset_dir / SPLIT_DIRS[split] / "_annotations.coco.json"


def predict_split(
    detector: SupportsDetect, split_data: CocoSplit, *, image_root: Path
) -> tuple[list[ImagePrediction], float]:
    """Run the detector over every image in a split. Returns predictions and seconds."""
    started = time.perf_counter()
    predictions: list[ImagePrediction] = []
    for image in split_data.images:
        detection = detector.detect(load_rgb(image_root / image.file_name))
        predictions.append(
            ImagePrediction(
                image_id=image.image_id,
                pred_boxes=detection.boxes_xyxy,
                pred_scores=detection.scores,
                gt_boxes=image.boxes_xyxy,
            )
        )
    return predictions, time.perf_counter() - started


def filter_by_confidence(
    predictions: list[ImagePrediction], threshold: float
) -> list[ImagePrediction]:
    """Keep only detections at or above `threshold`, ground truth untouched.

    Used to report precision and recall at the confidence a deployment would actually
    run at. Average precision is deliberately *not* computed from the filtered set:
    truncating the ranking chops off the tail of the recall axis and inflates AP.
    """
    filtered: list[ImagePrediction] = []
    for prediction in predictions:
        keep = np.asarray(prediction.pred_scores, dtype=np.float64) >= threshold
        filtered.append(
            ImagePrediction(
                image_id=prediction.image_id,
                pred_boxes=prediction.pred_boxes[keep],
                pred_scores=prediction.pred_scores[keep],
                gt_boxes=prediction.gt_boxes,
            )
        )
    return filtered


def score_predictions(
    predictions: list[ImagePrediction],
    *,
    run_id: str,
    split: str,
    iou_thresholds: tuple[float, ...],
    iou_thresholds_small: tuple[float, ...],
    coco_sweep: bool,
    operating_threshold: float | None = None,
    synthetic: bool = False,
) -> EvaluationReport:
    """Turn per-image predictions into a report. Pure: no model, no filesystem."""
    report = EvaluationReport(
        run_id=run_id, split=split, synthetic=synthetic, operating_threshold=operating_threshold
    )

    for threshold in iou_thresholds:
        report.overall[threshold] = evaluate_dataset(predictions, iou_threshold=threshold)

    if operating_threshold is not None:
        at_threshold = filter_by_confidence(predictions, operating_threshold)
        for threshold in iou_thresholds:
            report.operating_point[threshold] = evaluate_dataset(
                at_threshold, iou_threshold=threshold
            )
    if coco_sweep:
        report.overall_sweep = sweep_average_precision(predictions, thresholds=COCO_SWEEP)

    for bucket in SizeBucket:
        thresholds = iou_thresholds_small if bucket in SMALL_BUCKETS else iou_thresholds
        report.by_bucket[bucket] = {
            threshold: evaluate_dataset(predictions, iou_threshold=threshold, bucket=bucket)
            for threshold in thresholds
        }
        if coco_sweep:
            report.bucket_sweeps[bucket] = sweep_average_precision(
                predictions, thresholds=COCO_SWEEP, bucket=bucket
            )
    return report


@dataclass(frozen=True, slots=True)
class NegativeFootage:
    """What was found in a negatives directory, and how its duration is known."""

    images: tuple[Path, ...]
    videos: tuple[Path, ...]
    notes: tuple[str, ...] = ()

    @property
    def is_video(self) -> bool:
        return bool(self.videos)


def find_negative_footage(negatives_dir: Path) -> NegativeFootage:
    """Sort a negatives directory into videos and loose frames.

    Videos are preferred when both are present: a directory holding a clip and a few
    stray screenshots almost certainly meant the clip, and silently mixing the two would
    give a frame count with no coherent duration behind it.
    """
    if not negatives_dir.is_dir():
        raise ConfigError(f"eval.negatives_dir is not a directory: {negatives_dir}")

    videos = tuple(sorted(p for p in negatives_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES))
    images = tuple(sorted(p for p in negatives_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    if not videos and not images:
        raise ConfigError(
            f"nothing to measure in {negatives_dir}. Looked for videos "
            f"({', '.join(VIDEO_SUFFIXES)}) and frames ({', '.join(IMAGE_SUFFIXES)}). "
            "A false-alarm rate over zero frames is not a rate."
        )

    notes: list[str] = []
    if videos and images:
        notes.append(
            f"{len(images)} loose image(s) in {negatives_dir} were ignored: "
            f"{len(videos)} video(s) are present and mixing the two gives a frame count "
            "with no single duration behind it."
        )
        images = ()
    return NegativeFootage(images=images, videos=videos, notes=tuple(notes))


def measure_false_alarms(
    detector: SupportsDetect,
    negatives_dir: Path,
    *,
    fps: float | None,
    confidence_threshold: float,
) -> tuple[Any, tuple[str, ...]]:
    """Count every detection on footage asserted to contain no target.

    Accepts a directory of **videos** or of loose **frames**. For videos the frame rate
    comes from the container, which removes the worst failure mode of this metric: a
    per-hour rate scales linearly with fps, so a guessed one silently scales the headline
    number. For loose frames there is nothing to read it from, so `fps` is required and
    the caller must state it.

    Frames with no detections are kept on purpose: they are the denominator, and dropping
    them inflates the rate.
    """
    footage = find_negative_footage(negatives_dir)
    notes = list(footage.notes)

    if footage.is_video:
        from tayr.worker.pipeline import iter_frames
        from tayr.worker.probe import probe_video

        scores: list[Any] = []
        boxes: list[Any] = []
        durations: list[float] = []
        for video in footage.videos:
            # Probe BEFORE decoding, every time. Owner-supplied media is not a reason to
            # skip the check - a wrong path is far more likely than an attack here, and
            # the failure mode is identical.
            media = probe_video(video)
            if media.fps <= 0:
                raise ConfigError(
                    f"{video} reports fps={media.fps}. Rate per hour is frames/fps/3600, "
                    "so this cannot be turned into a rate; re-encode it or measure "
                    "against extracted frames with an explicit fps."
                )
            n_frames = 0
            for frame in iter_frames(video):
                detection = detector.detect(frame)
                scores.append(detection.scores)
                # Kept alongside the scores so the alarms can be bucketed by size. The
                # question "how big were the things it fired on" is what separates a
                # sensitivity problem from a discrimination one.
                boxes.append(detection.boxes_xyxy)
                n_frames += 1
            durations.append(n_frames / media.fps)
            notes.append(
                f"{video.name}: {n_frames} frame(s) at {media.fps:.3f} fps "
                f"= {n_frames / media.fps:.1f}s"
            )

        total_frames = len(scores)
        total_seconds = sum(durations)
        # An effective rate, so a set of clips at differing frame rates still yields one
        # honest duration rather than one arbitrarily chosen fps.
        effective_fps = total_frames / total_seconds if total_seconds > 0 else 0.0
        if fps is not None:
            notes.append(
                f"eval.negatives_fps={fps} was ignored: the frame rate came from the "
                f"containers themselves ({effective_fps:.3f} fps effective across "
                f"{len(footage.videos)} video(s)), which is the value that is actually true."
            )
        return (
            false_alarms_per_hour(
                scores,
                fps=effective_fps,
                confidence_threshold=confidence_threshold,
                pred_boxes_per_frame=boxes,
            ),
            tuple(notes),
        )

    if fps is None:
        raise ConfigError(
            f"{negatives_dir} holds loose frames, so there is no container to read a "
            "frame rate from. Set eval.negatives_fps: a per-hour rate is frames/fps/3600 "
            "and a guessed fps scales the headline number linearly and silently."
        )
    detections = [detector.detect(load_rgb(path)) for path in footage.images]
    notes.append(f"{len(detections)} loose frame(s) at a declared {fps} fps")
    return (
        false_alarms_per_hour(
            [d.scores for d in detections],
            fps=fps,
            confidence_threshold=confidence_threshold,
            pred_boxes_per_frame=[d.boxes_xyxy for d in detections],
        ),
        tuple(notes),
    )


#: Checkpoints RF-DETR writes, best first. `checkpoint_best_total.pth` is the one its
#: own BestModelCallback selects between the regular and EMA tracks.
#: [VERIFIED: rfdetr 1.9.4, rfdetr/detr.py:919-922 names all four]
CHECKPOINT_PREFERENCE: tuple[str, ...] = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "last_ema.pth",
)


def load_run_config(run_dir: Path) -> tuple[Config, Path]:
    """Rebuild the config a run was trained with, from its own manifest.

    `tayr eval` takes a run directory rather than a config file on purpose. Evaluating
    with a config that has drifted since training - a changed resolution, a different
    dataset path - produces a number that belongs to neither, and nothing in the output
    would show it. Reading the config back out of the manifest makes that impossible.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(
            f"no manifest.json in {run_dir}. `tayr eval` reads the config out of the "
            "run's own manifest, so the directory must be one `tayr train` produced."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{manifest_path} is not valid JSON: {exc}") from exc
    if "config" not in manifest:
        raise ConfigError(f"{manifest_path} has no 'config' key; it is not a run manifest.")
    return Config.model_validate(manifest["config"]), manifest_path


def resolve_run_checkpoint(run_dir: Path, *, name: str | None = None) -> tuple[Path, str]:
    """Pick a checkpoint from a run directory and hash it.

    The digest is computed here, from the run's own output, and recorded so a report
    names exactly which weights produced it. It is provenance, not authentication - it
    cannot detect tampering with a file this same process just hashed. The config-level
    `checkpoint_sha256` is the control that does that job, for weights that arrived from
    somewhere else.
    """
    from tayr.detection.rfdetr_backend import sha256_file

    if name is not None:
        path = run_dir / name
        if not path.is_file():
            raise ConfigError(f"checkpoint not found in {run_dir}: {name}")
        return path, sha256_file(path)

    for candidate in CHECKPOINT_PREFERENCE:
        path = run_dir / candidate
        if path.is_file():
            return path, sha256_file(path)

    raise ConfigError(
        f"no checkpoint found in {run_dir}. Looked for "
        f"{', '.join(CHECKPOINT_PREFERENCE)}. Has training finished an epoch?"
    )


def evaluate_split(
    cfg: Config,
    *,
    run_dir: Path,
    split: str = "test",
    detector: SupportsDetect | None = None,
    config_path: Path | None = None,
    repo: Path | None = None,
    synthetic: bool = False,
) -> SplitEvaluation:
    """Evaluate a trained detector on one split and write the report next to the run.

    `detector` is injectable so the scoring path can be exercised against a scripted
    stand-in without a model. When it is None a real RF-DETR is built from the config,
    and its checkpoint is checksum-verified before loading.
    """
    if cfg.train.dataset_dir is None:
        raise ConfigError("train.dataset_dir is not set, so there is no split to evaluate against.")
    annotation_path = split_annotation_path(cfg.train.dataset_dir, split)
    split_data = read_coco_split(annotation_path)

    seed_report = seed_everything(cfg.seed, strict_torch=detector is None)
    device: ResolvedDevice | None = None
    notes: list[str] = []

    if detector is None:
        from tayr.detection.rfdetr_backend import RFDetrDetector

        device = resolve_device(cfg.device)
        detector = RFDetrDetector(
            variant=cfg.detector.variant,
            checkpoint=cfg.detector.checkpoint,
            checkpoint_sha256=cfg.detector.checkpoint_sha256,
            device=device.device,
            # Collect at the lowest threshold any reported metric needs. Average
            # precision integrates the whole ranking, so cutting the tail off here would
            # truncate the recall axis and inflate AP.
            confidence_threshold=0.0,
            resolution=cfg.train.resolution,
        )
        notes.append(f"device: {device.note}")
        if cfg.detector.checkpoint is None:
            notes.append(
                "NO FINE-TUNED CHECKPOINT. This evaluated RF-DETR's published COCO "
                "weights, which were never trained on this project's data. The numbers "
                "describe that model, not a Tayr detector."
            )

    if not detector.is_real:
        synthetic = True
        notes.append(
            f"DETECTOR IS A STAND-IN ({detector.name}). Every number below is a test of "
            "the harness, not of a detector."
        )

    predictions, seconds = predict_split(detector, split_data, image_root=annotation_path.parent)
    report = score_predictions(
        predictions,
        run_id=run_dir.name,
        split=split,
        iou_thresholds=cfg.eval.iou_thresholds,
        iou_thresholds_small=cfg.eval.iou_thresholds_small,
        coco_sweep=cfg.eval.coco_sweep,
        operating_threshold=cfg.detector.confidence_threshold,
        synthetic=synthetic,
    )

    if cfg.eval.negatives_dir is not None:
        report.false_alarms, negative_notes = measure_false_alarms(
            detector,
            cfg.eval.negatives_dir,
            fps=cfg.eval.negatives_fps,
            confidence_threshold=cfg.detector.confidence_threshold,
        )
        notes.extend(negative_notes)

    if not split_data.has_video_grouping:
        notes.append(
            f"{annotation_path} has images without a `source_video` field, so it cannot "
            "be confirmed that this split is disjoint by video from training. Treat the "
            "numbers as an upper bound."
        )

    n_detections = sum(len(p.pred_scores) for p in predictions)
    notes.append(
        f"detector: {detector.name}; {len(predictions)} image(s), {n_detections} raw "
        f"detection(s) in {seconds:.1f}s"
    )
    degenerate = getattr(detector, "degenerate_boxes_dropped", 0)
    if degenerate:
        raw = getattr(detector, "raw_predictions", 0)
        worst = getattr(detector, "max_dropped_score", 0.0)
        share = f"{100.0 * degenerate / raw:.3f}% of {raw} raw prediction(s)" if raw else "?"
        operating = cfg.detector.confidence_threshold
        verdict = (
            "harmless: nothing dropped was ever going to be a detection"
            if worst < operating
            else "NOT harmless: a dropped box scored at or above the operating "
            "threshold, so the model is emitting confident degenerate boxes and these "
            "metrics need re-examining"
        )
        notes.append(
            f"{degenerate} zero-extent prediction(s) dropped before scoring - {share}. "
            f"Highest score among them {worst:.4f}, against an operating threshold of "
            f"{operating:.2f}: {verdict}. The denominator that matters is raw "
            "predictions, not true positives: a DETR head decodes 300 boxes for every "
            "image whatever it contains, so at a collection threshold of 0 the raw count "
            "is 300 x images. The mechanism is border clamping - see "
            "tayr.detection.rfdetr_backend._to_tayr_detection."
        )
    report.notes.extend(notes)

    manifest = build_manifest(
        run_id=run_dir.name,
        command=f"tayr eval --split {split}",
        config=cfg.to_dict(),
        seed_report=seed_report,
        config_path=config_path,
        repo=repo,
        synthetic=synthetic,
        notes=notes,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest.write(run_dir, filename=f"manifest.eval-{split}.json")
    report_path = run_dir / f"report-{split}.txt"
    report_path.write_text(report.render(), encoding="utf-8")

    return SplitEvaluation(
        report=report,
        n_images=len(predictions),
        n_detections=n_detections,
        seconds=seconds,
        device=device,
        manifest_path=manifest_path,
        report_path=report_path,
        notes=notes,
    )
