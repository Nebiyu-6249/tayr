"""RF-DETR behind Tayr's `Detector` protocol.

Everything in this module about the RF-DETR API was read out of the installed package
in the session that wrote it, not recalled. The citations say which file and which
lines, so the next person to touch this can re-check them in a minute:

  * `RFDETR.from_checkpoint(path, *, trust_checkpoint=False, **kwargs)`
    [VERIFIED: rfdetr 1.9.4, rfdetr/detr.py:450]
  * `RFDETR.predict(images, threshold=0.5, ..., include_source_image=True)` returns a
    `supervision.Detections` [VERIFIED: rfdetr/detr.py:2211-2223]
  * variant classes `RFDETRNano` / `RFDETRSmall` / `RFDETRMedium` / `RFDETRLarge`
    [VERIFIED: rfdetr/variants.py]
  * `RFDETR.__init__` calls `maybe_download_pretrain_weights()`, so constructing a
    variant with a non-None `pretrain_weights` performs a network fetch
    [VERIFIED: rfdetr/detr.py:397, 421-444]

## Why the checksum gate is here and not in a doc

`torch.load` unpickles, and unpickling executes. RF-DETR's own loader tries
`weights_only=True` first and only falls back to full pickle when `trust_checkpoint` is
set [VERIFIED: rfdetr/utilities/io.py:28-88], so this adapter never sets it - not for
convenience, not for a "trusted" file. But `weights_only=True` only stops code
execution; it does not tell you *which* weights you loaded. A run whose checkpoint
cannot be identified afterwards is not reproducible, so the file is hashed and compared
against the config before it is opened, and a mismatch raises.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from tayr.errors import ConfigError, DependencyUnavailableError, TayrError
from tayr.worker.detector import Detection

#: Tayr's variant names -> the class name exported by `rfdetr`.
#: [VERIFIED: rfdetr 1.9.4, rfdetr/variants.py]
VARIANT_CLASS_NAMES: Final[dict[str, str]] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

#: Read in this many bytes at a time when hashing a checkpoint. Checkpoints run to
#: hundreds of megabytes and reading one into memory to hash it is avoidable.
_HASH_CHUNK = 1024 * 1024


class CheckpointError(TayrError):
    """A checkpoint is missing, or its contents do not match the recorded checksum."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str) -> str:
    """Hash `path` and raise unless it matches `expected_sha256`.

    Returns the digest so the caller can record it. Case-insensitive on the expected
    value because hex digests get pasted from every kind of tool.
    """
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise CheckpointError(
            f"checkpoint {path} has sha256 {actual}, but the config records "
            f"{expected_sha256.strip().lower()}. Refusing to load it: either the file is "
            "not the one this run claims to use, or the config is stale. Both make the "
            "result unattributable."
        )
    return actual


def _variant_class(variant: str) -> Any:
    """Import and return the RF-DETR class for a variant name."""
    if variant not in VARIANT_CLASS_NAMES:
        raise ConfigError(
            f"unknown RF-DETR variant {variant!r}; expected one of "
            f"{', '.join(sorted(VARIANT_CLASS_NAMES))}"
        )
    rfdetr = _import_rfdetr()
    return getattr(rfdetr, VARIANT_CLASS_NAMES[variant])


def _import_rfdetr() -> Any:
    try:
        import rfdetr
    except ImportError as exc:
        raise DependencyUnavailableError(
            "rfdetr is required for the RF-DETR backend but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'"
        ) from exc
    return rfdetr


class RFDetrDetector:
    """RF-DETR as a Tayr `Detector`.

    Construct with a fine-tuned `checkpoint` (plus its recorded sha256) for a Tayr
    detector, or with `checkpoint=None` to get RF-DETR's published COCO weights. The
    second is a real detector and `is_real` is True either way - but it was never
    trained on this project's data, and `name` says so, because a COCO baseline number
    reported as a Tayr number would be a fabricated result.
    """

    def __init__(
        self,
        *,
        variant: str = "nano",
        checkpoint: Path | None = None,
        checkpoint_sha256: str | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        resolution: int | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ConfigError(f"confidence_threshold must be in [0, 1]; got {confidence_threshold}")
        if checkpoint is not None and checkpoint_sha256 is None:
            raise ConfigError(
                "a checkpoint may only be loaded together with its recorded sha256. "
                "See DetectorConfig.checkpoint_sha256."
            )

        self._variant = variant
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._resolution = resolution
        self._checkpoint_sha256: str | None = None
        self._degenerate_boxes_dropped = 0
        self._raw_predictions = 0
        self._max_dropped_score = 0.0

        rfdetr = _import_rfdetr()
        build_kwargs: dict[str, Any] = {"device": device}
        if resolution is not None:
            build_kwargs["resolution"] = resolution

        if checkpoint is not None and checkpoint_sha256 is not None:
            self._checkpoint_sha256 = verify_checkpoint(checkpoint, checkpoint_sha256)
            # trust_checkpoint stays False. With it False, RF-DETR loads under
            # weights_only=True and raises if that cannot succeed, rather than falling
            # back to a pickle load that would execute whatever is in the file.
            self._model = rfdetr.RFDETR.from_checkpoint(
                str(checkpoint), trust_checkpoint=False, **build_kwargs
            )
            self._provenance = f"checkpoint:{self._checkpoint_sha256[:12]}"
        else:
            # No checkpoint: RF-DETR downloads and MD5-checks its own published COCO
            # weights for this variant on construction.
            self._model = _variant_class(variant)(**build_kwargs)
            self._provenance = "coco-pretrained"

    @property
    def is_real(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"rfdetr-{self._variant}[{self._provenance}]"

    @property
    def checkpoint_sha256(self) -> str | None:
        """Recorded in the run manifest. None when running published COCO weights."""
        return self._checkpoint_sha256

    @property
    def degenerate_boxes_dropped(self) -> int:
        """Zero-extent predictions discarded so far. Read with `raw_predictions`."""
        return self._degenerate_boxes_dropped

    @property
    def raw_predictions(self) -> int:
        """Every box the model decoded, before any filtering.

        The denominator for the drop count, and the one that is easy to get wrong. A
        DETR head decodes `num_select` boxes for EVERY image regardless of content - 300
        for every RF-DETR variant - so at a collection threshold of 0 the raw count is
        `300 x n_images`, orders of magnitude above the number of true positives.
        Comparing drops against true positives instead makes a 0.07% rate look like 20%.
        """
        return self._raw_predictions

    @property
    def max_dropped_score(self) -> float:
        """The highest confidence among discarded boxes. The diagnostic that matters.

        A zero-area box has IoU 0 with everything, so it can never be a true positive
        and dropping it can only ever remove a false positive. That makes the count
        almost irrelevant and this number decisive: if it stays far below the operating
        threshold, nothing droppable was ever going to be a detection. If it approaches
        or exceeds that threshold, the model is emitting confident degenerate boxes and
        the metrics need re-examining.
        """
        return self._max_dropped_score

    @property
    def is_finetuned(self) -> bool:
        """False when this is RF-DETR's published COCO model rather than a Tayr one."""
        return self._checkpoint_sha256 is not None

    def detect(self, frame: npt.NDArray[np.uint8]) -> Detection:
        """Detect in one HxWx3 RGB frame."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ConfigError(
                f"expected an HxWx3 RGB frame; got shape {frame.shape}. RF-DETR takes "
                "RGB channel order, so a BGR frame from OpenCV must be converted first."
            )
        # include_source_image=False: the default attaches the frame to the result's
        # metadata, which at video rates keeps every decoded frame alive.
        result = self._model.predict(
            frame,
            threshold=self._confidence_threshold,
            include_source_image=False,
        )
        detection, discarded = _to_tayr_detection(result)
        self._raw_predictions += len(detection.scores) + discarded.count
        self._degenerate_boxes_dropped += discarded.count
        self._max_dropped_score = max(self._max_dropped_score, discarded.max_score)
        return detection


@dataclass(frozen=True, slots=True)
class DiscardedBoxes:
    """Zero-extent predictions removed at the boundary, and how confident they were."""

    count: int
    max_score: float


def _to_tayr_detection(result: Any) -> tuple[Detection, DiscardedBoxes]:
    """Convert a `supervision.Detections` into Tayr's `Detection`.

    Returns the detection and a summary of what was discarded.

    `predict` can return a list when given a list of images; this adapter always passes
    one frame, so a list here means the upstream contract changed and that is worth
    raising over rather than silently taking element zero.

    **Zero-extent boxes are dropped here, on purpose.** A DETR head decodes every one of
    its queries, and some of the low-confidence ones come back with zero width or height
    - `[212.8, 0.0, 374.3, 0.0]` was observed in the first real run against this adapter,
    a box clamped flat against the top edge. `tayr.geometry` refuses those rather than
    guessing a repair, which is right: everywhere else in this codebase a degenerate box
    means a converter bug. Here it does not, it means a model emitted a null query, and
    the boundary is the place to say so.

    Dropping is the honest treatment rather than a convenience. A zero-area box has IoU
    0 with everything, so it can never be a true positive; keeping it would only ever
    add a false positive for a prediction that makes no spatial claim at all.

    **The mechanism is border clamping**, measured rather than guessed: over 3,600 raw
    predictions from a trained checkpoint, every one of the 9 dropped boxes had height
    exactly 0.0 and sat on a frame edge `[VERIFIED: measured this session]`. A query
    predicts a box whose extent falls outside the frame, postprocessing clamps both
    edges to the same border, and the extent collapses. It is the ordinary fate of a
    few of the 300 decoded queries, not a fault in the resolution or the postprocessing.

    What is returned is the count *and the highest score among the dropped*, because the
    second is what decides whether any of it matters.
    """
    if isinstance(result, list):
        raise CheckpointError(
            "RF-DETR returned a list of predictions for a single frame. The upstream "
            "predict() contract has changed; re-read rfdetr/detr.py before trusting "
            "anything this adapter produces."
        )

    boxes = np.asarray(getattr(result, "xyxy", np.empty((0, 4))), dtype=np.float64).reshape(-1, 4)

    confidence = getattr(result, "confidence", None)
    if confidence is None:
        # supervision allows confidence=None. A detection with no score cannot be
        # ranked, and every metric in tayr.eval sorts by score, so this is fatal.
        raise CheckpointError(
            "RF-DETR returned detections with no confidence scores. Detection metrics "
            "rank by score, so scoreless boxes cannot be evaluated."
        )
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)

    class_id = getattr(result, "class_id", None)
    class_ids = None if class_id is None else np.asarray(class_id, dtype=np.int64).reshape(-1)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    discarded_scores = scores[~keep]

    return (
        Detection(
            boxes_xyxy=boxes[keep],
            scores=scores[keep],
            class_ids=None if class_ids is None else class_ids[keep],
        ),
        DiscardedBoxes(
            count=len(discarded_scores),
            max_score=float(discarded_scores.max()) if len(discarded_scores) else 0.0,
        ),
    )
