"""Demo tests.

The demo is a deliverable, so it is tested like one. A rule change that breaks it should
fail here, not in front of an audience.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tayr.agent.demo import DEFAULT_SCENARIOS, run_authorized_demo, run_demo
from tayr.agent.verdicts import Verdict
from tayr.errors import TayrError

pytest.importorskip("av", reason="PyAV is in the 'cv' extra")

from tests.test_worker_probe import make_video

SITES = Path("configs/sites/demo.yaml")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "scene.mp4", width=640, height=480, fps=30, n_frames=400)


class TestDemo:
    def test_produces_a_decision_per_scenario(self, video: Path, tmp_path: Path) -> None:
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        assert len(result.decisions) == len(DEFAULT_SCENARIOS)

    def test_each_scenario_triggers_its_intended_rule(self, video: Path, tmp_path: Path) -> None:
        """run_demo raises if a scenario applies an unexpected rule, so reaching here at
        all is the assertion. This names it explicitly."""
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        applied = {d.decision.rule_id for d in result.decisions}
        assert applied == {s.expect_rule for s in DEFAULT_SCENARIOS}

    def test_hover_escalates_and_flapping_dismisses(self, video: Path, tmp_path: Path) -> None:
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        by_rule = {d.decision.rule_id: d for d in result.decisions}
        assert by_rule["escalate.sustained_hover"].decision.verdict is Verdict.ESCALATE
        assert by_rule["dismiss.flapping_band"].decision.verdict is Verdict.DISMISS

    def test_hover_is_distinguished_from_flapping_by_power_not_frequency(
        self, video: Path, tmp_path: Path
    ) -> None:
        """The hovering target's dominant frequency lands inside the bird band; only its
        negligible power share keeps it from being dismissed. That distinction is the
        whole reason the power check exists."""
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        hover = next(
            d for d in result.decisions if d.decision.rule_id == "escalate.sustained_hover"
        )
        assert any("power share" in r for r in hover.decision.rationale)
        assert any("do not hover" in r for r in hover.decision.rationale)

    def test_authorized_flight_is_dismissed(self, video: Path, tmp_path: Path) -> None:
        """The dismissal that matters. Most detected drones are somebody's permitted
        flight, and suppressing them is the product."""
        result = run_authorized_demo(video, output_dir=tmp_path / "auth", site_registry_path=SITES)
        assert result.decisions[0].decision.verdict is Verdict.DISMISS
        assert result.decisions[0].decision.rule_id == "dismiss.authorized_flight"

    def test_everything_is_labelled_synthetic(self, video: Path, tmp_path: Path) -> None:
        """There is no trained detector. Every artefact must say so."""
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        assert all(d.synthetic for d in result.decisions)
        assert all(d.to_dict()["synthetic"] for d in result.decisions)
        posted = str(list(result.notifier.posted.values()))
        assert "Synthetic run" in posted

    def test_dismissals_go_to_the_audit_channel_not_the_alert_channel(
        self, video: Path, tmp_path: Path
    ) -> None:
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        channels = {p["channel"] for p in result.notifier.posted.values()}
        assert result.notifier.audit_channel in channels

    def test_decisions_are_written_to_disk(self, video: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        run_demo(video, output_dir=out, site_registry_path=SITES)
        assert (out / "decisions.json").is_file()
        assert list((out / "notifications").glob("*.json"))

    def test_runs_with_no_model_configured(self, video: Path, tmp_path: Path) -> None:
        """The default is no LLM, so the demo exercises degradation every time."""
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        assert all(d.prose is None for d in result.decisions)
        assert all(d.decision.rationale for d in result.decisions)

    def test_every_decision_is_reconstructable(self, video: Path, tmp_path: Path) -> None:
        result = run_demo(video, output_dir=tmp_path / "out", site_registry_path=SITES)
        for decision in result.decisions:
            record = decision.to_dict()
            assert record["rule_id"] and record["rationale"]
            assert len(record["tool_calls"]) >= 3
            assert all("arguments" in c and "result" in c for c in record["tool_calls"])
            assert decision.audit_hash()


class TestDemoFailsLoudly:
    def test_missing_video_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TayrError, match="demo video not found"):
            run_demo(tmp_path / "absent.mp4", output_dir=tmp_path, site_registry_path=SITES)

    def test_unknown_site_raises(self, video: Path, tmp_path: Path) -> None:
        with pytest.raises(TayrError, match="not in"):
            run_demo(video, output_dir=tmp_path, site_registry_path=SITES, site_id="nowhere")
