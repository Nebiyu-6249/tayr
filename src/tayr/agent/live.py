"""`tayr watch run`: the whole pipeline on one video, with a real detector.

This is `tayr watch demo` with the scripted detections taken out. Everything downstream
of detection is the same code the demo already ran - decode, tracking, motion features,
tool calls, verdict rules, notification - so what this adds is exactly one thing: the
boxes come from a trained model instead of from a list.

## The honesty label is derived, not declared

`synthetic` is `not detector.is_real`, computed once in `tayr.detection.factory` and
carried from there into the pipeline result, the tool context, the decision record, the
manifest and the notification. There is no parameter to set and no flag to remember,
because the failure this project cares about is a placeholder run being reported as a
real one, and that failure is always a human forgetting a flag.

The one thing a trained detector does **not** change is the classifier. `TrackSnapshot`
leaves `classifier_trained` False, so `analyze_track` still answers
`no_classifier_trained` - a different claim from "the classifier was unsure", and the
rules keep them apart. A detector that finds drones does not tell you whether one is a
bird, and nothing here may imply otherwise.

## Device

CPU by default and stated in the output, not assumed. Inference at ~1 image/s on a
laptop CPU is entirely adequate for a demo video; `--device` takes the same values as
every other command, and `auto` records that it fell back when it does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tayr.agent.llm import AgentLLM, UnavailableLLM
from tayr.agent.loop import TriageAgent
from tayr.agent.notify import LocalNotifier, Notifier
from tayr.agent.records import AgentDecision
from tayr.agent.site_config import SiteRegistry, load_site_registry
from tayr.agent.store import InMemoryStore
from tayr.agent.tools import ToolContext, TrackSnapshot, build_registry
from tayr.agent.verdicts import Verdict
from tayr.config import Config
from tayr.detection.factory import build_detector
from tayr.determinism import seed_everything
from tayr.devices import ResolvedDevice, resolve_device
from tayr.errors import ConfigError
from tayr.manifest import build_manifest
from tayr.render.annotate import RenderResult, render_annotated_video
from tayr.tracking.features import MIN_OBSERVATIONS
from tayr.worker.pipeline import PipelineResult, run_pipeline


@dataclass(frozen=True, slots=True)
class WatchRun:
    """One end-to-end run: what it saw, what it decided, and under what conditions."""

    decisions: list[AgentDecision]
    pipeline: PipelineResult
    device: ResolvedDevice
    detector_name: str
    checkpoint_sha256: str | None
    output_dir: Path
    manifest_path: Path
    seconds: float
    notes: list[str] = field(default_factory=list)
    render: RenderResult | None = None
    """The annotated video, when one was asked for. Off by default: rendering decodes
    and re-encodes the whole video a second time, which the headless path has no use
    for."""

    @property
    def synthetic(self) -> bool:
        """Derived from the detector, via the pipeline. Never set by hand."""
        return self.pipeline.synthetic

    @property
    def frames_per_second(self) -> float:
        return self.pipeline.frames_processed / self.seconds if self.seconds > 0 else 0.0

    def verdict_counts(self) -> dict[str, int]:
        return {v.value: sum(1 for d in self.decisions if d.decision.verdict is v) for v in Verdict}


def run_watch(
    video_path: Path,
    cfg: Config,
    *,
    output_dir: Path,
    site_registry_path: Path,
    site_id: str = "demo-north",
    llm: AgentLLM | None = None,
    notifier: Notifier | None = None,
    observed_at: datetime | None = None,
    config_path: Path | None = None,
    repo: Path | None = None,
    run_id: str | None = None,
    allow_stub: bool = False,
    render: bool = False,
) -> WatchRun:
    """Decode, detect, track, extract features, triage every usable track, notify."""
    if not video_path.is_file():
        raise ConfigError(f"video not found: {video_path}")

    registry: SiteRegistry = load_site_registry(site_registry_path)
    if registry.get(site_id) is None:
        raise ConfigError(f"site {site_id!r} is not in {site_registry_path}")

    seed_report = seed_everything(cfg.seed, strict_torch=not allow_stub)
    device = resolve_device(cfg.device)
    choice = build_detector(cfg.detector, device=device.device, allow_stub=allow_stub)

    output_dir.mkdir(parents=True, exist_ok=True)
    notifier = notifier or LocalNotifier(output_dir=output_dir / "notifications")
    store = InMemoryStore()
    agent = TriageAgent(llm=llm or UnavailableLLM(), registry=build_registry())
    moment = observed_at or datetime.now(UTC)

    started = time.perf_counter()
    result = run_pipeline(video_path, choice.detector, tracker_config=cfg.tracker)
    elapsed = time.perf_counter() - started

    notes = [f"device: {device.note}", *choice.notes, *result.notes]
    if device.fell_back:
        notes.append(
            "DEVICE FELL BACK TO CPU. Throughput below describes CPU execution and is "
            "not comparable with a GPU run."
        )

    decisions: list[AgentDecision] = []
    job_id = run_id or f"watch-{int(moment.timestamp())}"

    for track in result.tracks:
        features = result.features.get(track.track_id)
        track_id = f"{job_id}-t{track.track_id}"
        snapshot = TrackSnapshot(
            track_id=track_id,
            job_id=job_id,
            track_number=track.track_id,
            first_frame=track.observed_frames[0],
            last_frame=track.observed_frames[-1],
            n_observations=track.n_observations,
            median_pixels_on_target=(
                float(np.median(_pixels_on_target(track.observed_boxes)))
                if track.observed_boxes
                else 0.0
            ),
            fps=result.media.fps if result.media.fps > 0 else 30.0,
            features=features,
            observed_at=moment,
            # NOT set from the detector. A trained detector is not a trained classifier,
            # and `analyze_track` must keep answering `no_classifier_trained` until one
            # is actually fitted - which is a different claim from "the classifier was
            # unsure" and drives a different rule.
            classifier_trained=False,
        )
        context = ToolContext(
            site_id=site_id,
            tracks={track_id: snapshot},
            registry=registry,
            evidence_dir=output_dir / "evidence",
            now=moment,
            # Derived. `run_pipeline` computed it from `detector.is_real`.
            synthetic=result.synthetic,
        )
        decision = agent.triage(track_id, context, job_id=job_id)
        store.put_decision(decision)
        decisions.append(decision)

        if decision.decision.verdict is Verdict.DISMISS:
            notifier.post_dismissal(decision)
        else:
            notifier.post_escalation(decision)

    rendered: RenderResult | None = None
    if render:
        # After the verdicts, necessarily: a track's verdict needs its whole history, so
        # there is nothing to draw until the last frame has been through the pipeline.
        rendered = render_annotated_video(
            video_path,
            output_dir / "annotated.mp4",
            tracks=result.tracks,
            decisions=decisions,
            job_id=job_id,
            tracker_config=cfg.tracker,
        )
        notes.extend(rendered.notes)
        notes.append(
            f"annotated video: {rendered.frames_written} frame(s), "
            f"{rendered.boxes_drawn} box(es) drawn"
        )

    manifest = build_manifest(
        run_id=job_id,
        command="tayr watch run",
        config=cfg.to_dict(),
        seed_report=seed_report,
        config_path=config_path,
        repo=repo,
        synthetic=result.synthetic,
        notes=[*notes, f"detector: {choice.detector.name}"],
    )
    manifest_path = manifest.write(output_dir)

    (output_dir / "decisions.json").write_text(
        json.dumps([d.to_dict() for d in decisions], indent=2, default=str), encoding="utf-8"
    )

    return WatchRun(
        decisions=decisions,
        pipeline=result,
        device=device,
        detector_name=choice.detector.name,
        checkpoint_sha256=choice.checkpoint_sha256,
        output_dir=output_dir,
        manifest_path=manifest_path,
        seconds=elapsed,
        notes=notes,
        render=rendered,
    )


def _pixels_on_target(boxes: list[np.ndarray]) -> np.ndarray:
    from tayr.geometry import pixels_on_target

    return pixels_on_target(np.stack(boxes))


def observations_note(result: PipelineResult) -> str:
    """One line on how many tracks were long enough to carry motion features."""
    usable = sum(1 for t in result.tracks if t.n_observations >= MIN_OBSERVATIONS)
    return (
        f"{len(result.tracks)} track(s), {usable} with at least {MIN_OBSERVATIONS} "
        "observations and therefore motion features"
    )
