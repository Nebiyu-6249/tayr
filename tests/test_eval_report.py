"""Report rendering tests.

The report's job is to make omitting an inconvenient number harder than including it,
so these assert the warnings actually appear.
"""

from __future__ import annotations

import numpy as np

from tayr.eval.detection import DetectionCounts, evaluate_detection
from tayr.eval.false_alarms import false_alarms_per_hour
from tayr.eval.report import EvaluationReport
from tayr.geometry import SizeBucket


def _counts(side: float, bucket: SizeBucket, iou: float = 0.5) -> DetectionCounts:
    gt = np.array([[0.0, 0.0, side, side]])
    return evaluate_detection(gt.copy(), np.array([0.9]), gt, iou_threshold=iou, bucket=bucket)


class TestRender:
    def test_missing_false_alarm_rate_is_called_out(self) -> None:
        """Silence about false alarms would let mAP stand alone. It must not."""
        out = EvaluationReport(run_id="r", split="test").render()
        assert "NOT MEASURED" in out
        assert "describes half the system" in out

    def test_false_alarm_rate_is_shown_when_present(self) -> None:
        r = EvaluationReport(run_id="r", split="test")
        r.false_alarms = false_alarms_per_hour(
            [np.array([0.9])] + [np.array([])] * 99, fps=30, confidence_threshold=0.5
        )
        assert "/ hour" in r.render()

    def test_every_bucket_appears_even_when_empty(self) -> None:
        out = EvaluationReport(run_id="r", split="test").render()
        for bucket in SizeBucket:
            assert bucket.value in out

    def test_thin_bucket_is_flagged(self) -> None:
        r = EvaluationReport(run_id="r", split="test")
        r.by_bucket = {SizeBucket.TINY: {0.5: _counts(4.0, SizeBucket.TINY)}}
        assert "THIN" in r.render()

    def test_synthetic_runs_are_labelled_loudly(self) -> None:
        out = EvaluationReport(run_id="r", split="test", synthetic=True).render()
        assert "SYNTHETIC DATA" in out
        assert "no number below describes real-world performance" in out

    def test_protocol_is_stated_in_the_report_itself(self) -> None:
        """Someone reading only the output must be able to tell it is not
        pycocotools-comparable."""
        out = EvaluationReport(run_id="r", split="test").render()
        assert "grouped by source video" in out
        assert "101-point" in out
        assert "not directly comparable" in out

    def test_both_iou_thresholds_render_for_a_bucket(self) -> None:
        r = EvaluationReport(run_id="r", split="test")
        r.by_bucket = {
            SizeBucket.TINY: {
                0.25: _counts(4.0, SizeBucket.TINY, iou=0.25),
                0.5: _counts(4.0, SizeBucket.TINY, iou=0.5),
            }
        }
        out = r.render()
        assert "AP@0.25" in out and "AP@0.50" in out
