"""RF-DETR adapter: the checkpoint gate and the boundary conversion.

Both are testable without torch. The conversion takes a duck-typed stand-in for
`supervision.Detections` rather than a real one, which keeps these tests runnable on a
machine with no cv extra - and keeps them honest about what the adapter actually reads
off the object: three attributes, nothing more.
"""

from __future__ import annotations

#: sha256 of b"weights" - computed here rather than pasted, so the test cannot drift.
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tayr.detection.rfdetr_backend import (
    VARIANT_CLASS_NAMES,
    CheckpointError,
    _to_tayr_detection,
    _variant_class,
    sha256_file,
    verify_checkpoint,
)
from tayr.errors import ConfigError
from tests.cv_extra import requires_cv_extra

PAYLOAD = b"weights"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoint_best_total.pth"
    path.write_bytes(PAYLOAD)
    return path


def detections(xyxy: Any, confidence: Any = None, class_id: Any = None) -> SimpleNamespace:
    """A stand-in with the three attributes the adapter reads."""
    return SimpleNamespace(xyxy=np.asarray(xyxy), confidence=confidence, class_id=class_id)


class TestCheckpointGate:
    def test_a_matching_digest_returns_it(self, checkpoint: Path) -> None:
        assert verify_checkpoint(checkpoint, DIGEST) == DIGEST

    def test_case_and_whitespace_in_the_recorded_digest_are_tolerated(
        self, checkpoint: Path
    ) -> None:
        assert verify_checkpoint(checkpoint, f"  {DIGEST.upper()}  ") == DIGEST

    def test_a_mismatch_refuses_to_load(self, checkpoint: Path) -> None:
        with pytest.raises(CheckpointError, match="Refusing to load it"):
            verify_checkpoint(checkpoint, "0" * 64)

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="checkpoint not found"):
            verify_checkpoint(tmp_path / "nope.pth", DIGEST)

    def test_hashing_is_chunked_and_matches_hashlib(self, tmp_path: Path) -> None:
        """Checkpoints run to hundreds of MB; the hash must not read one into memory."""
        big = tmp_path / "big.pth"
        blob = bytes(range(256)) * 8192
        big.write_bytes(blob)
        assert sha256_file(big) == hashlib.sha256(blob).hexdigest()


class TestVariantResolution:
    @pytest.mark.parametrize("variant", sorted(VARIANT_CLASS_NAMES))
    def test_every_configured_variant_exists_upstream(self, variant: str) -> None:
        pytest.importorskip("rfdetr", reason="rfdetr is in the cv extra")
        assert _variant_class(variant).__name__ == VARIANT_CLASS_NAMES[variant]

    def test_an_unknown_variant_lists_the_valid_ones(self) -> None:
        with pytest.raises(ConfigError, match="unknown RF-DETR variant"):
            _variant_class("enormous")

    def test_the_deprecated_base_variant_is_not_offered(self) -> None:
        """RFDETRBase is scheduled for removal in 2.0.0; a dissertation should not use it."""
        assert "base" not in VARIANT_CLASS_NAMES


class TestBoundaryConversion:
    def test_boxes_and_scores_survive(self) -> None:
        detection, dropped = _to_tayr_detection(
            detections([[0.0, 0.0, 10.0, 10.0]], confidence=np.array([0.7]), class_id=np.array([1]))
        )
        assert detection.boxes_xyxy.shape == (1, 4)
        assert detection.scores.tolist() == [0.7]
        assert detection.class_ids is not None
        assert detection.class_ids.tolist() == [1]
        assert dropped.count == 0

    def test_zero_extent_boxes_are_dropped_and_counted(self) -> None:
        """Observed in the first real run: a DETR query decoded flat against the top edge.

        It has IoU 0 with everything, so it can never be a true positive - keeping it
        would only ever contribute a false positive for a prediction that makes no
        spatial claim.
        """
        detection, dropped = _to_tayr_detection(
            detections(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [212.8, 0.0, 374.3, 0.0],  # zero height
                    [5.0, 5.0, 5.0, 40.0],  # zero width
                ],
                confidence=np.array([0.7, 0.2, 0.1]),
                class_id=np.array([1, 1, 1]),
            )
        )
        assert dropped.count == 2
        assert len(detection.boxes_xyxy) == 1
        assert detection.scores.tolist() == [0.7]
        assert detection.class_ids is not None
        assert detection.class_ids.tolist() == [1]

    def test_scoreless_detections_are_fatal_rather_than_scored_as_one(self) -> None:
        """supervision allows confidence=None. Every metric here ranks by score."""
        with pytest.raises(CheckpointError, match="no confidence scores"):
            _to_tayr_detection(detections([[0.0, 0.0, 1.0, 1.0]], confidence=None))

    def test_missing_class_ids_are_allowed(self) -> None:
        detection, _ = _to_tayr_detection(
            detections([[0.0, 0.0, 4.0, 4.0]], confidence=np.array([0.5]), class_id=None)
        )
        assert detection.class_ids is None

    def test_a_list_result_raises_instead_of_taking_element_zero(self) -> None:
        """One frame in, one result out. A list means the upstream contract moved."""
        with pytest.raises(CheckpointError, match="contract has changed"):
            _to_tayr_detection([detections([[0.0, 0.0, 1.0, 1.0]], confidence=np.array([0.5]))])

    def test_an_empty_result_is_not_an_error(self) -> None:
        detection, dropped = _to_tayr_detection(
            detections(np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0))
        )
        assert len(detection.boxes_xyxy) == 0
        assert dropped.count == 0


@requires_cv_extra
class TestCheckpointLoadRefusesPickle:
    """The first externally-supplied checkpoint made this rule real rather than stated.

    CLAUDE.md 3 says `torch.load` executes pickle and that internal loads use
    `weights_only=True`. That had been asserted from reading RF-DETR's source. These run
    it against a checkpoint carrying a payload that writes a file when unpickled: if the
    file appears, code executed during load.
    """

    def hostile_checkpoint(self, tmp_path: Path) -> tuple[Path, Path]:
        """A checkpoint whose payload writes a marker. Returns (checkpoint, marker)."""
        import torch

        marker = tmp_path / "CODE-EXECUTED-DURING-LOAD"

        class Payload:
            def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
                # Deliberately benign and confined to pytest's tmp_path: the point is to
                # prove the payload is live, so that the refusal tests below are not
                # quietly passing against an inert file.
                return (Path.write_text, (marker, "the loader executed this"))

        path = tmp_path / "hostile.pth"
        torch.save({"model": {"w": torch.tensor([1.0])}, "args": {}, "extra": Payload()}, path)
        return path, marker

    def test_the_payload_is_live_so_the_refusals_below_mean_something(self, tmp_path: Path) -> None:
        """A guard whose test could pass against an inert payload is not a guard."""
        from rfdetr.utilities.io import _safe_torch_load

        path, marker = self.hostile_checkpoint(tmp_path)
        assert not marker.exists()
        _safe_torch_load(path, trust=True)
        assert marker.exists(), "the payload did not execute even with trust=True"

    def test_weights_only_refuses_and_executes_nothing(self, tmp_path: Path) -> None:
        import torch

        path, marker = self.hostile_checkpoint(tmp_path)
        with pytest.raises(Exception, match=r"[Ww]eights only"):
            torch.load(path, map_location="cpu", weights_only=True)
        assert not marker.exists()

    def test_the_loader_the_adapter_reaches_refuses(self, tmp_path: Path) -> None:
        """`trust_checkpoint=False` is the flag the adapter never flips."""
        from rfdetr.utilities.io import _safe_torch_load

        path, marker = self.hostile_checkpoint(tmp_path)
        with pytest.raises(RuntimeError, match="Failed to safely load"):
            _safe_torch_load(path, trust=False)
        assert not marker.exists()

    def test_the_adapter_refuses_a_hostile_checkpoint(self, tmp_path: Path) -> None:
        from tayr.detection.rfdetr_backend import RFDetrDetector, sha256_file

        path, marker = self.hostile_checkpoint(tmp_path)
        with pytest.raises(Exception, match=r"(?i)safely load|weights only|checkpoint"):
            RFDetrDetector(
                variant="nano",
                checkpoint=path,
                checkpoint_sha256=sha256_file(path),
                device="cpu",
            )
        assert not marker.exists()


class TestDropAccounting:
    """The count is nearly irrelevant; the highest dropped score is what decides."""

    def test_raw_predictions_is_the_denominator_not_true_positives(self) -> None:
        detection, discarded = _to_tayr_detection(
            detections(
                [[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 20.0, 0.0]],
                confidence=np.array([0.9, 0.01]),
            )
        )
        # Raw = kept + discarded. A DETR head decodes num_select boxes for every image
        # regardless of content, so this is orders of magnitude above the TP count.
        assert len(detection.scores) + discarded.count == 2

    def test_the_highest_dropped_score_is_reported(self) -> None:
        _, discarded = _to_tayr_detection(
            detections(
                [[5.0, 0.0, 20.0, 0.0], [1.0, 1.0, 1.0, 9.0]],
                confidence=np.array([0.004, 0.002]),
            )
        )
        assert discarded.count == 2
        assert discarded.max_score == pytest.approx(0.004)

    def test_no_drops_reports_a_zero_max_rather_than_none(self) -> None:
        """None would have to be special-cased by every comparison against a threshold."""
        _, discarded = _to_tayr_detection(
            detections([[0.0, 0.0, 4.0, 4.0]], confidence=np.array([0.5]))
        )
        assert discarded.count == 0
        assert discarded.max_score == 0.0
