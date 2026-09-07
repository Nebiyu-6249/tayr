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
        assert dropped == 0

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
        assert dropped == 2
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
        assert dropped == 0
