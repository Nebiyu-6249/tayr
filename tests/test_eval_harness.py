"""Dataset-level evaluation: pooling, bucketing, the operating point, and the report.

Every test here runs against a scripted detector rather than a model. That is the point:
a metric bug and a model bug look identical in a final number, so the metric path is
exercised against inputs whose correct answer is known by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tayr.config import Config
from tayr.datasets.coco import read_coco_split
from tayr.errors import ConfigError, GeometryError
from tayr.eval.detection import (
    COCO_SWEEP,
    ImagePrediction,
    evaluate_dataset,
    evaluate_detection,
    sweep_average_precision,
)
from tayr.eval.harness import (
    SPLIT_DIRS,
    filter_by_confidence,
    score_predictions,
    split_annotation_path,
)
from tayr.eval.report import EvaluationReport
from tayr.geometry import SizeBucket

BOX = [10.0, 10.0, 30.0, 30.0]  # 20px on target -> MEDIUM
TINY_BOX = [10.0, 10.0, 15.0, 15.0]  # 5px on target -> TINY


def prediction(
    image_id: int, preds: list[list[float]], scores: list[float], gt: list[list[float]]
) -> ImagePrediction:
    return ImagePrediction(
        image_id=image_id,
        pred_boxes=np.asarray(preds, dtype=np.float64).reshape(-1, 4),
        pred_scores=np.asarray(scores, dtype=np.float64),
        gt_boxes=np.asarray(gt, dtype=np.float64).reshape(-1, 4),
    )


class TestPooling:
    def test_pooled_ranking_is_not_the_mean_of_per_image_scores(self) -> None:
        """The distinction this whole function exists for.

        Image A: a confident hit. Image B: a low-confidence hit plus a false positive
        that outranks it. Per image both score 1.0 and 0.5; pooled, A's hit and B's false
        positive interleave by confidence and the answer is neither the mean nor either
        value.
        """
        images = [
            prediction(1, [BOX], [0.9], [BOX]),
            prediction(2, [[100.0, 100.0, 120.0, 120.0], BOX], [0.8, 0.2], [BOX]),
        ]
        per_image = [
            evaluate_detection(i.pred_boxes, i.pred_scores, i.gt_boxes).average_precision
            for i in images
        ]
        pooled = evaluate_dataset(images).average_precision
        assert pooled != pytest.approx(sum(per_image) / len(per_image))
        assert 0.0 < pooled < 1.0

    def test_counts_are_summed_across_images(self) -> None:
        images = [prediction(i, [BOX], [0.9], [BOX]) for i in range(4)]
        counts = evaluate_dataset(images)
        assert counts.true_positives == 4
        assert counts.n_eligible_gt == 4
        assert counts.false_positives == 0
        assert counts.average_precision == pytest.approx(1.0)

    def test_frames_with_no_ground_truth_still_contribute_false_positives(self) -> None:
        """Empty frames are the denominator of the precision story; dropping them lies."""
        images = [
            prediction(1, [BOX], [0.9], [BOX]),
            prediction(2, [[50.0, 50.0, 70.0, 70.0]], [0.95], []),
        ]
        counts = evaluate_dataset(images)
        assert counts.true_positives == 1
        assert counts.false_positives == 1

    def test_an_empty_split_raises_rather_than_scoring_zero(self) -> None:
        """ "AP 0.0" and "there was nothing to evaluate" must not render identically."""
        with pytest.raises(GeometryError, match="no images to evaluate"):
            evaluate_dataset([])


class TestBucketing:
    def test_out_of_bucket_ground_truth_is_ignored_not_counted_against(self) -> None:
        """COCO ignore semantics, applied across images.

        Without them, a correct detection of a 20px object would be a false positive in
        the TINY bucket, and small-bucket precision would be meaningless.
        """
        images = [
            prediction(1, [TINY_BOX], [0.9], [TINY_BOX]),
            prediction(2, [BOX], [0.9], [BOX]),
        ]
        tiny = evaluate_dataset(images, bucket=SizeBucket.TINY)
        assert tiny.n_eligible_gt == 1
        assert tiny.true_positives == 1
        assert tiny.false_positives == 0

    def test_a_bucket_with_no_ground_truth_reports_zero_eligible(self) -> None:
        images = [prediction(1, [BOX], [0.9], [BOX])]
        assert evaluate_dataset(images, bucket=SizeBucket.TINY).n_eligible_gt == 0


class TestSweep:
    def test_the_sweep_has_all_ten_coco_thresholds(self) -> None:
        images = [prediction(1, [BOX], [0.9], [BOX])]
        sweep = sweep_average_precision(images)
        assert set(sweep) == set(COCO_SWEEP)
        assert len(COCO_SWEEP) == 10

    def test_a_loose_box_scores_at_low_iou_and_not_at_high(self) -> None:
        loose = [12.0, 12.0, 32.0, 32.0]  # shifted by 2px on a 20px box
        images = [prediction(1, [loose], [0.9], [BOX])]
        sweep = sweep_average_precision(images)
        assert sweep[0.5] == pytest.approx(1.0)
        assert sweep[0.95] == pytest.approx(0.0)


class TestOperatingPoint:
    def test_filtering_keeps_ground_truth_so_recall_still_falls(self) -> None:
        """Dropping the ground truth alongside the detections would hide the misses."""
        images = [prediction(1, [BOX], [0.1], [BOX])]
        filtered = filter_by_confidence(images, 0.5)
        assert len(filtered[0].pred_scores) == 0
        assert len(filtered[0].gt_boxes) == 1
        assert evaluate_dataset(filtered).recall == 0.0

    def test_average_precision_is_computed_over_the_unfiltered_ranking(self) -> None:
        """Truncating the ranking chops the recall axis and inflates AP."""
        images = [prediction(1, [BOX], [0.1], [BOX])]
        report = score_predictions(
            images,
            run_id="r",
            split="test",
            iou_thresholds=(0.5,),
            iou_thresholds_small=(0.25, 0.5),
            coco_sweep=False,
            operating_threshold=0.5,
        )
        assert report.overall[0.5].average_precision == pytest.approx(1.0)
        assert report.operating_point[0.5].recall == 0.0
        assert report.operating_point[0.5].false_negatives == 1

    def test_the_report_names_the_threshold_it_used(self) -> None:
        images = [prediction(1, [BOX], [0.9], [BOX])]
        rendered = score_predictions(
            images,
            run_id="r",
            split="test",
            iou_thresholds=(0.5,),
            iou_thresholds_small=(0.25, 0.5),
            coco_sweep=False,
            operating_threshold=0.25,
        ).render()
        assert "deployment threshold conf>=0.25" in rendered


class TestReportHonesty:
    def test_a_partial_sweep_is_not_averaged_into_a_map(self) -> None:
        """ "mAP@0.50:0.95" over four thresholds is a different quantity with the same name."""
        report = EvaluationReport(run_id="r", split="test")
        report.overall_sweep = {0.5: 0.9, 0.55: 0.8, 0.6: 0.7}
        assert report.map_50_95 is None
        assert "the sweep ran only 3 of 10 thresholds" in report.render()

    def test_a_full_sweep_is_averaged_and_labelled(self) -> None:
        report = EvaluationReport(run_id="r", split="test")
        report.overall_sweep = dict.fromkeys(COCO_SWEEP, 0.5)
        report.overall = {}
        assert report.map_50_95 == pytest.approx(0.5)

    def test_small_buckets_get_the_low_iou_threshold(self) -> None:
        images = [prediction(1, [TINY_BOX], [0.9], [TINY_BOX])]
        report = score_predictions(
            images,
            run_id="r",
            split="test",
            iou_thresholds=(0.5,),
            iou_thresholds_small=(0.25, 0.5),
            coco_sweep=False,
        )
        assert set(report.by_bucket[SizeBucket.TINY]) == {0.25, 0.5}
        assert set(report.by_bucket[SizeBucket.LARGE]) == {0.5}


class TestSplitPaths:
    def test_val_maps_onto_the_directory_rf_detr_actually_reads(self) -> None:
        """RF-DETR's roboflow loader calls the directory `valid`, not `val`."""
        assert SPLIT_DIRS["val"] == "valid"
        assert split_annotation_path(Path("/ds"), "val") == Path("/ds/valid/_annotations.coco.json")

    def test_an_unknown_split_lists_the_valid_ones(self) -> None:
        with pytest.raises(ConfigError, match="unknown split"):
            split_annotation_path(Path("/ds"), "holdout")


class TestCocoReader:
    def write_split(self, tmp_path: Path, *, with_video: bool = True) -> Path:
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "a.jpg",
                    "width": 64,
                    "height": 64,
                    **({"source_video": "v0"} if with_video else {}),
                },
                {
                    "id": 2,
                    "file_name": "b.jpg",
                    "width": 64,
                    "height": 64,
                    **({"source_video": "v0"} if with_video else {}),
                },
            ],
            # COCO bbox is [x, y, w, h]. Image 2 has no annotations on purpose.
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "drone"}],
        }
        path = tmp_path / "_annotations.coco.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_xywh_is_converted_to_xyxy(self, tmp_path: Path) -> None:
        split = read_coco_split(self.write_split(tmp_path))
        assert split.images[0].boxes_xyxy[0].tolist() == [10.0, 10.0, 30.0, 30.0]

    def test_images_with_no_annotations_are_kept(self, tmp_path: Path) -> None:
        """They are where false positives appear; dropping them removes the negatives."""
        split = read_coco_split(self.write_split(tmp_path))
        assert len(split.images) == 2
        assert len(split.images[1].boxes_xyxy) == 0
        assert split.n_boxes == 1

    def test_video_grouping_is_reported_when_absent(self, tmp_path: Path) -> None:
        assert read_coco_split(self.write_split(tmp_path)).has_video_grouping is True
        assert (
            read_coco_split(self.write_split(tmp_path, with_video=False)).has_video_grouping
            is False
        )

    def test_a_non_coco_file_is_rejected_by_name(self, tmp_path: Path) -> None:
        path = tmp_path / "_annotations.coco.json"
        path.write_text(json.dumps({"images": []}), encoding="utf-8")
        with pytest.raises(ConfigError, match="not a COCO detection file"):
            read_coco_split(path)

    def test_malformed_json_is_reported_with_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "_annotations.coco.json"
        path.write_text("{oh no", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            read_coco_split(path)


def test_config_refuses_negatives_without_a_frame_rate(tmp_path: Path) -> None:
    """A guessed fps scales false-alarms-per-hour linearly and silently."""
    with pytest.raises(ValueError, match="negatives_fps is required"):
        Config.model_validate({"eval": {"negatives_dir": str(tmp_path)}})
