"""Detection metric tests.

Every case here has an answer computable by hand, because a subtly wrong AP silently
inflates every number in the dissertation and nothing downstream would flag it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.errors import GeometryError
from tayr.eval.detection import (
    average_precision,
    evaluate_detection,
    match_detections,
)
from tayr.geometry import SizeBucket


def box(x: float, y: float, side: float) -> list[float]:
    return [x, y, x + side, y + side]


class TestMatching:
    def test_perfect_detection(self) -> None:
        gt = np.array([box(0, 0, 10)])
        r = match_detections(gt.copy(), np.array([0.9]), gt, iou_threshold=0.5)
        assert r.is_true_positive.tolist() == [True]
        assert r.n_eligible_gt == 1

    def test_duplicate_detection_of_one_object_is_a_false_positive(self) -> None:
        """Each ground truth is matched at most once. The second detection of the same
        object is a false positive, not a second true positive."""
        gt = np.array([box(0, 0, 10)])
        preds = np.array([box(0, 0, 10), box(0, 0, 10)])
        r = match_detections(preds, np.array([0.9, 0.8]), gt, iou_threshold=0.5)
        assert r.is_true_positive.tolist() == [True, False]

    def test_higher_confidence_detection_wins_the_match(self) -> None:
        """Matching is greedy in descending confidence order, not input order."""
        gt = np.array([box(0, 0, 10)])
        preds = np.array([box(2, 0, 10), box(0, 0, 10)])
        r = match_detections(preds, np.array([0.3, 0.95]), gt, iou_threshold=0.5)
        assert r.is_true_positive.tolist() == [False, True]

    def test_below_threshold_is_not_a_match(self) -> None:
        gt = np.array([box(0, 0, 10)])
        pred = np.array([box(9, 0, 10)])  # 10% overlap
        r = match_detections(pred, np.array([0.9]), gt, iou_threshold=0.5)
        assert r.is_true_positive.tolist() == [False]

    def test_empty_inputs(self) -> None:
        empty = np.empty((0, 4))
        assert match_detections(empty, np.array([]), empty, iou_threshold=0.5).n_eligible_gt == 0
        r = match_detections(empty, np.array([]), np.array([box(0, 0, 10)]), iou_threshold=0.5)
        assert r.n_eligible_gt == 1

    def test_mismatched_scores_length_raises(self) -> None:
        with pytest.raises(GeometryError, match="score"):
            match_detections(
                np.array([box(0, 0, 10)]),
                np.array([0.5, 0.6]),
                np.array([box(0, 0, 10)]),
                iou_threshold=0.5,
            )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_iou_threshold_raises(self, bad: float) -> None:
        with pytest.raises(GeometryError, match="iou_threshold"):
            match_detections(
                np.array([box(0, 0, 10)]),
                np.array([0.5]),
                np.array([box(0, 0, 10)]),
                iou_threshold=bad,
            )


class TestAreaRangeIgnoreSemantics:
    """COCO ignore semantics. Getting this wrong makes small-bucket precision
    meaningless, because every correct large-object detection would score as a
    small-bucket false positive."""

    def test_detection_of_out_of_range_gt_is_ignored_not_penalised(self) -> None:
        gt = np.array([box(0, 0, 4), box(100, 100, 64)])  # TINY + LARGE
        preds = np.array([box(0, 0, 4), box(100, 100, 64)])
        counts = evaluate_detection(
            preds, np.array([0.9, 0.9]), gt, iou_threshold=0.5, bucket=SizeBucket.TINY
        )
        assert counts.n_eligible_gt == 1
        assert counts.true_positives == 1
        assert counts.false_positives == 0  # the LARGE detection is ignored, not a FP
        assert counts.average_precision == pytest.approx(1.0)

    def test_genuine_false_positive_still_counts_in_bucket(self) -> None:
        gt = np.array([box(0, 0, 4)])
        preds = np.array([box(0, 0, 4), box(500, 500, 4)])
        counts = evaluate_detection(
            preds, np.array([0.9, 0.8]), gt, iou_threshold=0.5, bucket=SizeBucket.TINY
        )
        assert (counts.true_positives, counts.false_positives) == (1, 1)

    def test_eligible_gt_is_preferred_over_ignored_gt(self) -> None:
        """A detection overlapping both an in-range and an out-of-range object should
        be a true positive, not silently ignored."""
        gt = np.array([box(0, 0, 4), box(0, 0, 64)])
        counts = evaluate_detection(
            np.array([box(0, 0, 4)]),
            np.array([0.9]),
            gt,
            iou_threshold=0.5,
            bucket=SizeBucket.TINY,
        )
        assert counts.true_positives == 1

    def test_bucket_with_no_ground_truth(self) -> None:
        gt = np.array([box(0, 0, 64)])
        counts = evaluate_detection(
            np.array([box(0, 0, 64)]),
            np.array([0.9]),
            gt,
            iou_threshold=0.5,
            bucket=SizeBucket.TINY,
        )
        assert counts.n_eligible_gt == 0
        assert counts.average_precision == 0.0
        assert counts.recall == 0.0

    def test_bucket_boundaries_match_geometry_module(self) -> None:
        """A 16px box must fall in MEDIUM in both size_bucket() and BUCKET_RANGES,
        or bucketed metrics would disagree with the census."""
        gt = np.array([box(0, 0, 16)])
        medium = evaluate_detection(gt.copy(), np.array([0.9]), gt, bucket=SizeBucket.MEDIUM)
        small = evaluate_detection(gt.copy(), np.array([0.9]), gt, bucket=SizeBucket.SMALL)
        assert medium.n_eligible_gt == 1
        assert small.n_eligible_gt == 0


class TestAveragePrecision:
    def test_all_correct_gives_one(self) -> None:
        tp = np.array([True, True, True])
        ap = average_precision(tp, np.zeros(3, bool), np.array([0.9, 0.8, 0.7]), 3)
        assert ap == pytest.approx(1.0)

    def test_nothing_correct_gives_zero(self) -> None:
        tp = np.array([False, False])
        assert average_precision(tp, np.zeros(2, bool), np.array([0.9, 0.8]), 2) == 0.0

    def test_hand_computed_half_recall(self) -> None:
        """One TP then one FP, two GT. Recall reaches 0.5 at precision 1.0 and never
        rises again, so AP = 0.5 * 1.0 = 0.5."""
        tp = np.array([True, False])
        ap = average_precision(tp, np.zeros(2, bool), np.array([0.9, 0.8]), 2)
        assert ap == pytest.approx(0.5)

    def test_hand_computed_interleaved(self) -> None:
        """TP, FP, TP with 2 GT.
        After d1: R=0.5 P=1.0.  After d2: R=0.5 P=0.5.  After d3: R=1.0 P=2/3.
        Monotonic-from-right precision at R=0.5 becomes max(1.0, 2/3) = 1.0.
        AP = 0.5*1.0 + 0.5*(2/3) = 0.8333..."""
        tp = np.array([True, False, True])
        ap = average_precision(tp, np.zeros(3, bool), np.array([0.9, 0.8, 0.7]), 3 - 1)
        assert ap == pytest.approx(0.5 + 0.5 * (2 / 3))

    def test_missed_ground_truth_caps_recall(self) -> None:
        """One perfect detection but four GT: recall cannot exceed 0.25."""
        tp = np.array([True])
        assert average_precision(tp, np.zeros(1, bool), np.array([0.9]), 4) == pytest.approx(0.25)

    def test_ignored_detections_are_excluded_from_the_curve(self) -> None:
        tp = np.array([True, False])
        ignored = np.array([False, True])
        ap = average_precision(tp, ignored, np.array([0.9, 0.8]), 1)
        assert ap == pytest.approx(1.0)

    def test_no_eligible_gt_returns_zero(self) -> None:
        assert average_precision(np.array([True]), np.zeros(1, bool), np.array([0.9]), 0) == 0.0

    def test_confidence_order_changes_ap(self) -> None:
        """AP depends on ranking: a confident FP ahead of a TP costs precision."""
        good = average_precision(
            np.array([True, False]), np.zeros(2, bool), np.array([0.9, 0.1]), 1
        )
        bad = average_precision(np.array([False, True]), np.zeros(2, bool), np.array([0.9, 0.1]), 1)
        assert good == pytest.approx(1.0)
        assert bad == pytest.approx(0.5)
        assert bad < good


class TestSmallObjectRegime:
    def test_three_pixel_slip_on_an_eight_pixel_box_is_a_miss_at_iou_half(self) -> None:
        """The executable justification for reporting AP@0.25 in the small buckets."""
        gt = np.array([box(0, 0, 8)])
        pred = np.array([[3.0, 0.0, 11.0, 8.0]])
        at_50 = evaluate_detection(pred, np.array([0.9]), gt, iou_threshold=0.5)
        at_25 = evaluate_detection(pred, np.array([0.9]), gt, iou_threshold=0.25)
        assert at_50.true_positives == 0
        assert at_25.true_positives == 1

    def test_same_slip_on_a_large_box_is_harmless(self) -> None:
        """Identical absolute error, opposite verdict. This asymmetry is the reason
        bucketed reporting is mandatory rather than a nicety."""
        gt = np.array([box(0, 0, 80)])
        pred = np.array([[3.0, 0.0, 83.0, 80.0]])
        assert evaluate_detection(pred, np.array([0.9]), gt, iou_threshold=0.5).true_positives == 1
