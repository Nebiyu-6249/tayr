"""End-to-end demo: video in, verdicts out.

One command, so a rehearsal cannot drift from what gets recorded.

**Everything this produces is labelled synthetic**, because there is no trained
detector. The demo uses the existing scripted-detection path over a genuinely decoded
video, which yields real tracks with real motion features and a `synthetic=True` flag
that travels into the decision record, the notification and the rendered page. Nothing
here describes real-world detection performance, and the output says so at every level.

What is real: the decode, the tracking, the motion features, the tool calls, the
verdict rules, the audit records, and the notification bodies. What is placeholder: the
detections that started it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from tayr.agent.llm import AgentLLM, UnavailableLLM
from tayr.agent.loop import TriageAgent
from tayr.agent.notify import LocalNotifier
from tayr.agent.records import AgentDecision
from tayr.agent.site_config import SiteRegistry, load_site_registry
from tayr.agent.store import InMemoryStore
from tayr.agent.tools import ToolContext, TrackSnapshot, build_registry
from tayr.agent.verdicts import Verdict
from tayr.config import TrackerConfig
from tayr.errors import TayrError
from tayr.tracking.features import MIN_OBSERVATIONS, extract_features
from tayr.worker.detector import Detection, ScriptedDetector
from tayr.worker.pipeline import run_pipeline


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """One synthetic target with a motion signature chosen to exercise a rule."""

    label: str
    kind: str
    """hover | flapper | transit"""

    side_px: float
    expect_rule: str
    """The rule this scenario is built to trigger. Checked at the end, so a rule change
    that breaks the demo fails loudly rather than at the podium."""


DEFAULT_SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario("station-keeping object", "hover", 14.0, "escalate.sustained_hover"),
    DemoScenario("flapping flight", "flapper", 22.0, "dismiss.flapping_band"),
)


def _scenario_detections(scenario: DemoScenario, n_frames: int, fps: float) -> list[Detection]:
    """Per-frame boxes for one scenario. SYNTHETIC by construction."""
    t = np.arange(n_frames, dtype=np.float64)
    if scenario.kind == "hover":
        # Station-keeping jitter must be NOISE, not a sinusoid. A sinusoid - however
        # small - is a clean spectral peak, and the oscillation power-share check
        # correctly reads it as a real oscillation, which is not what hovering looks
        # like. This is the distinction the rule exists to make.
        rng = np.random.default_rng(4242)
        cx = np.full(n_frames, 320.0) + rng.normal(0.0, 0.12, n_frames)
        cy = np.full(n_frames, 200.0) + rng.normal(0.0, 0.12, n_frames)
    elif scenario.kind == "flapper":
        cx = 60.0 + t * 3.0
        cy = 300.0 + 9.0 * np.sin(2 * np.pi * 5.0 * t / fps)
    else:
        cx, cy = 40.0 + t * 6.0, np.full(n_frames, 400.0)

    half = scenario.side_px / 2.0
    return [
        Detection(
            np.array([[cx[i] - half, cy[i] - half, cx[i] + half, cy[i] + half]], dtype=np.float64),
            np.array([0.88], dtype=np.float64),
        )
        for i in range(n_frames)
    ]


@dataclass
class DemoResult:
    decisions: list[AgentDecision]
    notifier: LocalNotifier
    output_dir: Path

    @property
    def counts(self) -> dict[str, int]:
        return {v.value: sum(1 for d in self.decisions if d.decision.verdict is v) for v in Verdict}


def run_demo(
    video_path: Path,
    *,
    output_dir: Path,
    site_registry_path: Path,
    site_id: str = "demo-north",
    scenarios: tuple[DemoScenario, ...] = DEFAULT_SCENARIOS,
    llm: AgentLLM | None = None,
    observed_at: datetime | None = None,
) -> DemoResult:
    """Decode a video, build tracks per scenario, triage each one.

    Each scenario runs the pipeline separately so its motion signature is unambiguous.
    A single mixed video would produce tracks whose verdicts depend on association luck,
    which makes a poor demo and a worse test.
    """
    if not video_path.is_file():
        raise TayrError(f"demo video not found: {video_path}")

    registry: SiteRegistry = load_site_registry(site_registry_path)
    if registry.get(site_id) is None:
        raise TayrError(f"site {site_id!r} is not in {site_registry_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    notifier = LocalNotifier(output_dir=output_dir / "notifications")
    store = InMemoryStore()
    agent = TriageAgent(llm=llm or UnavailableLLM(), registry=build_registry())

    # Outside every authorized window in the demo registry, so the authorization rule
    # does not silently absorb every scenario. The authorized-flight demo is a separate,
    # explicit step in the runbook.
    moment = observed_at or datetime(2026, 9, 3, 22, 30, tzinfo=UTC)

    decisions: list[AgentDecision] = []
    for index, scenario in enumerate(scenarios):
        result = run_pipeline(
            video_path,
            ScriptedDetector(_scenario_detections(scenario, 400, 30.0)),
            tracker_config=TrackerConfig(min_hits=3, max_age=20, centre_distance_factor=2.0),
        )
        if not result.tracks:
            raise TayrError(
                f"scenario {scenario.label!r} produced no tracks; the demo cannot report "
                "a verdict on a track that does not exist"
            )

        track = max(result.tracks, key=lambda t: t.n_observations)
        boxes = np.stack(track.observed_boxes)
        frames = np.array(track.observed_frames, dtype=np.int64)
        feats = (
            extract_features(boxes, frames, fps=result.media.fps or 30.0)
            if track.n_observations >= MIN_OBSERVATIONS
            else None
        )

        track_id = f"demo-{index}-{scenario.kind}"
        snapshot = TrackSnapshot(
            track_id=track_id,
            job_id="demo-job",
            track_number=track.track_id,
            first_frame=track.observed_frames[0],
            last_frame=track.observed_frames[-1],
            n_observations=track.n_observations,
            median_pixels_on_target=float(feats.median_pixels_on_target) if feats else 0.0,
            fps=result.media.fps or 30.0,
            features=feats,
            observed_at=moment,
        )
        context = ToolContext(
            site_id=site_id,
            tracks={track_id: snapshot},
            registry=registry,
            evidence_dir=output_dir / "evidence",
            now=moment,
            # Derived from the detector via the pipeline, not asserted here. It is True
            # in this demo because ScriptedDetector.is_real is False - but if that ever
            # changes, the label follows it instead of staying stale. A hand-set flag is
            # how a placeholder run gets reported as a real one.
            synthetic=result.synthetic,
        )

        decision = agent.triage(track_id, context, job_id="demo-job")
        store.put_decision(decision)
        decisions.append(decision)

        if decision.decision.verdict is Verdict.DISMISS:
            notifier.post_dismissal(decision)
        else:
            notifier.post_escalation(decision)

        if decision.decision.rule_id != scenario.expect_rule:
            # Loud, not silent: a rule change that breaks the demo should fail here and
            # not in front of an audience.
            raise TayrError(
                f"scenario {scenario.label!r} expected rule {scenario.expect_rule!r} but the "
                f"agent applied {decision.decision.rule_id!r}. Either the scenario or the "
                "rules changed; reconcile them before demoing."
            )

    (output_dir / "decisions.json").write_text(
        json.dumps([d.to_dict() for d in decisions], indent=2, default=str), encoding="utf-8"
    )
    return DemoResult(decisions=decisions, notifier=notifier, output_dir=output_dir)


def run_authorized_demo(
    video_path: Path,
    *,
    output_dir: Path,
    site_registry_path: Path,
    site_id: str = "demo-north",
    llm: AgentLLM | None = None,
) -> DemoResult:
    """The dismissal that matters: an authorized flight, inside its filed window.

    This is the product. Most detected drones are somebody's permitted flight, and the
    value of the system is in suppressing them without a human ever being paged.
    """
    registry = load_site_registry(site_registry_path)
    site = registry.get(site_id)
    if site is None or not site.authorized_flights:
        raise TayrError(f"site {site_id!r} has no authorized flight to demonstrate against")

    inside_window = site.authorized_flights[0].starts_at + timedelta(minutes=30)
    return run_demo(
        video_path,
        output_dir=output_dir,
        site_registry_path=site_registry_path,
        site_id=site_id,
        scenarios=(
            DemoScenario("authorized survey flight", "hover", 18.0, "dismiss.authorized_flight"),
        ),
        llm=llm,
        observed_at=inside_window,
    )
