"""Detection metrics: matching, average precision, and pixels-on-target bucketing.

PROTOCOL, stated explicitly because a metric whose definition is vague is a metric that
cannot be compared to anything:

**Matching.** Detections are sorted by descending confidence and matched greedily to
ground-truth boxes with IoU >= threshold. Each ground-truth box is matched at most once;
later detections of the same object become false positives. This is the standard COCO
matching rule.

**Area ranges follow COCO's ignore semantics.** When evaluating a pixels-on-target
bucket, ground truth outside the bucket is marked *ignore* rather than deleted. A
detection matched to an ignored ground truth is dropped entirely - neither true nor
false positive. Deleting out-of-bucket ground truth instead would turn every correct
detection of a large object into a false positive in the small-object bucket, and make
small-bucket precision meaningless.

**Average precision uses all-point interpolation** - the exact area under the
precision-recall curve after making precision monotonically decreasing.

> IMPORTANT FOR THE WRITEUP: `pycocotools` reports 101-point interpolated AP, which is
> a slightly different quantity. Numbers from this module are therefore **not** directly
> comparable to published COCO-protocol figures. If a comparison against a published
> number is made, say which interpolation each side used, in the same sentence. See
> CLAUDE.md "Evaluation honesty".

**Small boxes.** IoU is savage at small sizes: on an 8x8 box a 3px offset already scores
as a miss at IoU 0.5. That is inside annotation noise, so `evaluate_detection` is
expected to be called at IoU 0.25 as well for the TINY and SMALL buckets, and both
reported. See docs/RESEARCH.md 6.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tayr.errors import GeometryError
from tayr.geometry import FloatArray, SizeBucket, iou, pixels_on_target

# (lo, hi] in pixels-on-target, matching tayr.geometry.size_bucket's half-open edges.
BUCKET_RANGES: dict[SizeBucket, tuple[float, float]] = {
    SizeBucket.TINY: (0.0, 8.0),
    SizeBucket.SMALL: (8.0, 16.0),
    SizeBucket.MEDIUM: (16.0, 32.0),
    SizeBucket.LARGE: (32.0, float("inf")),
}


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Per-detection outcome, in descending-confidence order."""

    is_true_positive: npt.NDArray[np.bool_]
    is_ignored: npt.NDArray[np.bool_]
    """Matched to an out-of-range ground truth; excluded from both TP and FP."""
    matched_gt_index: npt.NDArray[np.int64]
    """Index of the matched ground truth, or -1."""
    n_eligible_gt: int
    """Ground-truth boxes inside the evaluated area range. The recall denominator."""


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    """Aggregate outcome of an evaluation."""

    true_positives: int
    false_positives: int
    false_negatives: int
    n_eligible_gt: int
    average_precision: float
    iou_threshold: float
    bucket: SizeBucket | None = None

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.n_eligible_gt if self.n_eligible_gt else 0.0


def _in_range(boxes: FloatArray, area_range: tuple[float, float] | None) -> npt.NDArray[np.bool_]:
    if area_range is None:
        return np.ones(len(boxes), dtype=np.bool_)
    lo, hi = area_range
    pot = pixels_on_target(boxes)
    return (pot >= lo) & (pot < hi)


def match_detections(
    pred_boxes: FloatArray,
    pred_scores: npt.NDArray[np.float64],
    gt_boxes: FloatArray,
    *,
    iou_threshold: float,
    area_range: tuple[float, float] | None = None,
) -> MatchResult:
    """Greedily match detections to ground truth by descending confidence."""
    if not 0.0 < iou_threshold <= 1.0:
        raise GeometryError(f"iou_threshold must be in (0, 1]; got {iou_threshold}")

    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)

    if len(pred_scores) != len(pred_boxes):
        raise GeometryError(f"{len(pred_boxes)} prediction box(es) but {len(pred_scores)} score(s)")

    gt_eligible = _in_range(gt_boxes, area_range) if len(gt_boxes) else np.zeros(0, dtype=np.bool_)
    n_eligible = int(gt_eligible.sum())

    n_pred = len(pred_boxes)
    tp = np.zeros(n_pred, dtype=np.bool_)
    ignored = np.zeros(n_pred, dtype=np.bool_)
    matched_idx = np.full(n_pred, -1, dtype=np.int64)

    if n_pred == 0 or len(gt_boxes) == 0:
        return MatchResult(tp, ignored, matched_idx, n_eligible)

    order = np.argsort(-pred_scores, kind="stable")
    iou_matrix = iou(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=np.bool_)

    for pred_i in order:
        candidates = iou_matrix[pred_i].copy()
        candidates[gt_taken] = -1.0
        # Prefer eligible ground truth: a detection that could match either an
        # in-range or an out-of-range object should count as a true positive rather
        # than being silently ignored.
        eligible_scores = np.where(gt_eligible, candidates, -1.0)
        best_eligible = int(np.argmax(eligible_scores)) if len(eligible_scores) else -1

        if best_eligible >= 0 and eligible_scores[best_eligible] >= iou_threshold:
            tp[pred_i] = True
            matched_idx[pred_i] = best_eligible
            gt_taken[best_eligible] = True
            continue

        best_any = int(np.argmax(candidates))
        if candidates[best_any] >= iou_threshold:
            # Only reachable for an out-of-range ground truth, which is ignored.
            ignored[pred_i] = True
            matched_idx[pred_i] = best_any
            gt_taken[best_any] = True

    return MatchResult(tp, ignored, matched_idx, n_eligible)


def average_precision(
    is_true_positive: npt.NDArray[np.bool_],
    is_ignored: npt.NDArray[np.bool_],
    pred_scores: npt.NDArray[np.float64],
    n_eligible_gt: int,
) -> float:
    """All-point interpolated area under the precision-recall curve.

    Returns 0.0 when there is no eligible ground truth: with nothing to find, average
    precision is undefined, and returning 0 keeps a bucket with no objects from
    inflating a mean. Callers that average across buckets must skip empty ones.
    """
    if n_eligible_gt == 0:
        return 0.0

    keep = ~is_ignored
    tp = is_true_positive[keep]
    scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)[keep]
    if len(tp) == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    tp = tp[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)

    recall = tp_cum / n_eligible_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Make precision monotonically decreasing from the right, then integrate.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def evaluate_detection(
    pred_boxes: FloatArray,
    pred_scores: npt.NDArray[np.float64],
    gt_boxes: FloatArray,
    *,
    iou_threshold: float = 0.5,
    bucket: SizeBucket | None = None,
) -> DetectionCounts:
    """Evaluate one set of detections against one set of ground truth.

    Pass `bucket` to restrict to a pixels-on-target range using COCO ignore semantics.
    """
    area_range = BUCKET_RANGES[bucket] if bucket is not None else None
    match = match_detections(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold, area_range=area_range
    )

    tp = int(match.is_true_positive.sum())
    fp = int((~match.is_true_positive & ~match.is_ignored).sum())
    ap = average_precision(
        match.is_true_positive, match.is_ignored, pred_scores, match.n_eligible_gt
    )

    return DetectionCounts(
        true_positives=tp,
        false_positives=fp,
        false_negatives=match.n_eligible_gt - tp,
        n_eligible_gt=match.n_eligible_gt,
        average_precision=ap,
        iou_threshold=iou_threshold,
        bucket=bucket,
    )
