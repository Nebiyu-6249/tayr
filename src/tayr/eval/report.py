"""Assemble and render an evaluation report.

The report's job is to make the honest version of a result easier to produce than the
flattering one. Concretely: every bucket is shown even when empty, mAP@0.25 is printed
alongside mAP@0.5 in the small buckets, and false-alarm rate sits next to mAP rather
than in a separate section where it could quietly be omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tayr.eval.detection import COCO_SWEEP, DetectionCounts
from tayr.eval.false_alarms import FalseAlarmRate
from tayr.geometry import SizeBucket

# Below this many ground-truth boxes, a bucket's AP is too noisy to quote as a point
# estimate. It is still shown - suppressing it would hide a hole in the test set.
_THIN_BUCKET = 30


@dataclass(slots=True)
class EvaluationReport:
    """One evaluation run's results."""

    run_id: str
    split: str
    overall: dict[float, DetectionCounts] = field(default_factory=dict)
    """IoU threshold -> counts, over all sizes."""
    by_bucket: dict[SizeBucket, dict[float, DetectionCounts]] = field(default_factory=dict)
    """bucket -> IoU threshold -> counts."""
    overall_sweep: dict[float, float] = field(default_factory=dict)
    """IoU threshold -> AP over the COCO 0.50:0.05:0.95 sweep, all sizes. The mean of
    these is mAP@0.50:0.95, and it is a property rather than a stored number so it can
    never disagree with the values it is supposed to summarise."""

    bucket_sweeps: dict[SizeBucket, dict[float, float]] = field(default_factory=dict)

    operating_point: dict[float, DetectionCounts] = field(default_factory=dict)
    """IoU threshold -> counts among detections at or above `operating_threshold`.

    Average precision integrates the whole ranking, so it has to be computed over
    detections collected at confidence 0. The precision that falls out of that is not a
    number anyone would deploy against - it is one over the length of the tail. Precision
    and recall are therefore reported separately, at the confidence the config actually
    says to run at."""

    operating_threshold: float | None = None
    false_alarms: FalseAlarmRate | None = None
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _mean_ap(sweep: dict[float, float]) -> float | None:
        """Mean of a sweep, or None when it is incomplete.

        Refusing to average a partial sweep is the point: "mAP@0.50:0.95" computed over
        four thresholds is a different quantity with the same name, and reporting it
        under that label would be a fabricated comparison.
        """
        if set(sweep) != set(COCO_SWEEP):
            return None
        return sum(sweep.values()) / len(sweep)

    @property
    def map_50_95(self) -> float | None:
        """COCO's primary metric over all sizes, or None if the sweep was not run."""
        return self._mean_ap(self.overall_sweep)

    def bucket_map_50_95(self, bucket: SizeBucket) -> float | None:
        return self._mean_ap(self.bucket_sweeps.get(bucket, {}))

    def render(self) -> str:
        lines: list[str] = []

        if self.synthetic:
            lines += [
                "!" * 66,
                "!! SYNTHETIC DATA - no number below describes real-world performance",
                "!" * 66,
                "",
            ]

        lines += [f"Evaluation: {self.run_id} [{self.split}]", "=" * 66, ""]

        if self.overall:
            lines.append("  Overall (all sizes), integrated over the full ranking:")
            for thr in sorted(self.overall):
                c = self.overall[thr]
                lines.append(
                    f"    AP@{thr:.2f}        {c.average_precision:.4f}   "
                    f"({c.n_eligible_gt} GT boxes)"
                )

        # Outside the `overall` block on purpose: a sweep that ran but could not be
        # averaged must say so even when nothing else was computed. Nested under the
        # block above, an incomplete sweep rendered as silence.
        if self.overall_sweep:
            if not self.overall:
                lines.append("  Overall (all sizes), integrated over the full ranking:")
            headline = self.map_50_95
            if headline is not None:
                lines.append(f"    mAP@0.50:0.95  {headline:.4f}   (COCO primary metric)")
            else:
                lines.append(
                    f"    mAP@0.50:0.95  not reported - the sweep ran only "
                    f"{len(self.overall_sweep)} of {len(COCO_SWEEP)} thresholds"
                )

        if self.overall or self.overall_sweep:
            lines.append("")

        if self.operating_point and self.operating_threshold is not None:
            lines.append(f"  At the deployment threshold conf>={self.operating_threshold:.2f}:")
            for thr in sorted(self.operating_point):
                c = self.operating_point[thr]
                lines.append(
                    f"    IoU@{thr:.2f}  P {c.precision:.3f}  R {c.recall:.3f}   "
                    f"TP {c.true_positives} FP {c.false_positives} FN {c.false_negatives}"
                )
            lines.append("")

        lines.append("  By pixels-on-target  <- the research question lives here:")
        for bucket in (SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE):
            per_iou = self.by_bucket.get(bucket, {})
            if not per_iou:
                lines.append(f"    {bucket.value:10s}  (not evaluated)")
                continue

            n_gt = next(iter(per_iou.values())).n_eligible_gt
            if n_gt == 0:
                lines.append(f"    {bucket.value:10s}  no ground truth in this bucket")
                continue

            parts = [
                f"AP@{thr:.2f} {per_iou[thr].average_precision:.4f}" for thr in sorted(per_iou)
            ]
            bucket_map = self.bucket_map_50_95(bucket)
            if bucket_map is not None:
                parts.append(f"mAP@.50:.95 {bucket_map:.4f}")
            flag = f"   [THIN: only {n_gt} GT boxes]" if n_gt < _THIN_BUCKET else ""
            lines.append(f"    {bucket.value:10s}  {'  '.join(parts)}   (n={n_gt}){flag}")
        lines.append("")

        if self.false_alarms is not None:
            lines.append("  Drone-free footage:")
            lines += [f"    {ln}" for ln in self.false_alarms.render().splitlines()]
        else:
            lines.append(
                "  Drone-free footage: NOT MEASURED. mAP without a false-alarm rate "
                "describes half the system. Set eval.negatives_dir to a directory of "
                "drone-free clips (*.mp4 preferred - the frame rate is read from the "
                "container; loose frames need eval.negatives_fps as well)."
            )
        lines.append("")

        lines += [
            "  Protocol:",
            "    - splits grouped by source video",
            "    - AP is all-point interpolated, NOT pycocotools' 101-point form, so these",
            "      numbers are not directly comparable to published COCO figures",
            "    - small buckets report AP@0.25 because IoU@0.5 on an 8px box is inside "
            "annotation noise",
        ]

        if self.notes:
            lines += ["", "  NOTES:"] + [f"    - {n}" for n in self.notes]

        return "\n".join(lines)
