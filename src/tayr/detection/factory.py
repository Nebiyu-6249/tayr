"""One place that decides whether a run has a real detector.

The honesty label in this project is derived, never declared: `PipelineResult.synthetic`
is `not detector.is_real` and nothing sets it by hand. That only holds if there is a
single place where a detector is chosen, so every caller inherits the same answer -
otherwise the next call site to construct a `StubDetector()` inline is the one that
reports a placeholder run as real.

**A stub is never a fallback.** `build_detector` raises when it has no checkpoint unless
the caller has said, in that call, that a stand-in is acceptable. Handing back a stub to
a caller that wanted a detector would satisfy the type and quietly produce a run with no
detections in it - which is exactly the degraded path CLAUDE.md 1.4 forbids, and it would
be labelled correctly while still being useless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tayr.config import DetectorConfig
from tayr.errors import ConfigError
from tayr.worker.detector import Detector, StubDetector

#: Prefix on the note that says a checkpoint's own training run consumed placeholder or
#: synthetic data. `synthetic` on this run stays False - the detector really is a trained
#: model, and that is what the flag means - but a number produced by weights fitted to
#: synthetic data does not describe real-world performance either, and CLAUDE.md 1.3
#: requires that to be said in every report rather than inferred from a run id.
TRAINED_ON_SYNTHETIC = "WEIGHTS TRAINED ON SYNTHETIC DATA."


@dataclass(frozen=True, slots=True)
class DetectorChoice:
    """The detector a run got, and everything a manifest needs to say about it."""

    detector: Detector
    checkpoint_sha256: str | None
    notes: tuple[str, ...] = ()

    @property
    def synthetic(self) -> bool:
        """Derived, always. The one definition of the honesty label."""
        return not self.detector.is_real


def build_detector(
    cfg: DetectorConfig,
    *,
    device: str,
    allow_stub: bool = False,
) -> DetectorChoice:
    """Construct the detector a config asks for.

    `allow_stub` is a per-call decision rather than a config key on purpose: whether a
    stand-in is acceptable depends on what the caller is doing, not on the dataset. The
    worker accepts one so a job can still exercise decode and tracking; `tayr watch run`
    does not, because a triage decision from a detector that finds nothing is not a
    demonstration of anything.
    """
    if cfg.backend != "rfdetr":
        raise ConfigError(
            f"detector.backend={cfg.backend!r} has no implementation. Only 'rfdetr' is "
            "implemented; see docs/RESEARCH.md section 14 for why."
        )

    if cfg.checkpoint is None:
        if not allow_stub:
            raise ConfigError(
                "no detector checkpoint. Pass --checkpoint <path> (and, once you have "
                "recorded it, --checkpoint-sha256) or set detector.checkpoint in the "
                "config. Refusing to substitute a stand-in: it would find nothing, and a "
                "verdict reached from no detections demonstrates nothing."
            )
        return DetectorChoice(
            detector=StubDetector(),
            checkpoint_sha256=None,
            notes=(
                "NO TRAINED DETECTOR. A stand-in that detects nothing is in use, so this "
                "run is marked synthetic and no number in it describes real-world "
                "performance. Decode, tracking and motion features are still real.",
            ),
        )

    from tayr.detection.rfdetr_backend import RFDetrDetector, sha256_file

    # Computed whether or not it was supplied: a run has to be able to say which weights
    # produced it. Supplying it additionally *verifies* - see the note below.
    digest = sha256_file(cfg.checkpoint)
    notes: list[str] = [
        f"detector checkpoint sha256 {digest}",
        _training_provenance(cfg.checkpoint),
    ]
    if cfg.checkpoint_sha256 is None:
        notes.append(
            "CHECKSUM RECORDED BUT NOT VERIFIED. The digest above was computed from the "
            "file just now, so it identifies what ran; it cannot detect that the file "
            "was replaced before this run. Pin it with --checkpoint-sha256 to make the "
            "next run check rather than merely record."
        )

    detector = RFDetrDetector(
        variant=cfg.variant,
        checkpoint=cfg.checkpoint,
        # RFDetrDetector verifies against whichever value it is given. Passing the
        # computed digest when none was configured keeps one load path rather than two.
        checkpoint_sha256=cfg.checkpoint_sha256 or digest,
        device=device,
        confidence_threshold=cfg.confidence_threshold,
    )
    return DetectorChoice(detector=detector, checkpoint_sha256=digest, notes=tuple(notes))


def _training_provenance(checkpoint: Path) -> str:
    """What the checkpoint's own training run said about its data.

    `synthetic` on an inference run is `not detector.is_real`, which answers "were these
    detections produced by a model or by a stand-in". It does not answer "were these
    weights fitted to real footage" - and a detector trained on placeholder data is a real
    model producing numbers that describe nothing, which is the same failure arriving
    through a different door.

    `tayr train` writes a manifest beside its checkpoints, so the answer is usually one
    file away. When it is not, the note says the provenance is unknown rather than
    implying it is fine: an absent manifest is not evidence of real training data.
    """
    manifest = checkpoint.parent / "manifest.json"
    if not manifest.is_file():
        return (
            "TRAINING PROVENANCE UNKNOWN. No manifest.json beside this checkpoint, so "
            "what it was trained on cannot be established from the file alone."
        )
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Not swallowed, and not fatal either: an unreadable sibling file says nothing
        # about the checkpoint, but claiming provenance was checked would be false.
        return f"TRAINING PROVENANCE UNREADABLE. {manifest} could not be parsed: {exc}"

    run_id = recorded.get("run_id", "?")
    commit = str(recorded.get("git_commit", "?"))[:12]
    if recorded.get("synthetic") is True:
        return (
            f"{TRAINED_ON_SYNTHETIC} Run {run_id!r} (commit {commit}) recorded "
            "synthetic=true, so these weights were fitted to placeholder data. Detections "
            "below are real model output and describe no real-world performance."
        )
    return f"weights from run {run_id!r} (commit {commit}), trained on non-synthetic data"
