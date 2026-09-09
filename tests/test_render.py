"""The annotated video, and the one property it must not violate.

A red box is the most persuasive thing this project produces, and today it almost always
means "the system could not tell", not "the system found a drone". These tests exist to
stop the picture claiming more than `decisions.json` does - which is a different failure
from a wrong box, and a much easier one to ship without noticing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tayr.agent.records import AgentDecision, VerdictDecision
from tayr.agent.verdicts import UNCERTAIN_REASONS, Attention, Uncertainty, Verdict
from tayr.config import TrackerConfig
from tayr.errors import ConfigError
from tayr.render.annotate import (
    FORMING_COLOUR,
    UNCERTAINTY_CAPTIONS,
    VERDICT_COLOURS,
    TrackOverlay,
    _boxes_by_frame,
    _caveat_for,
    build_overlays,
    render_annotated_video,
)
from tayr.tracking.kalman import KalmanBoxFilter
from tayr.tracking.tracker import Track
from tests.cv_extra import requires_cv_extra

JOB = "test-run"


def decision(
    track_number: int,
    verdict: Verdict,
    *,
    uncertainty: Uncertainty = Uncertainty.NONE,
    rule_id: str = "rule.test",
) -> AgentDecision:
    return AgentDecision(
        track_id=f"{JOB}-t{track_number}",
        job_id=JOB,
        site_id="demo-north",
        decision=VerdictDecision(
            verdict=verdict,
            attention=Attention.ROUTINE,
            uncertainty=uncertainty,
            rule_id=rule_id,
        ),
    )


def track(track_number: int, *, n: int = 6, start: int = 0) -> Track:
    """A track with `n` observations on consecutive frames, drifting slowly."""
    box = np.array([100.0, 100.0, 120.0, 120.0])
    t = Track(track_id=track_number, filter=KalmanBoxFilter(box))
    t.observed_frames.clear()
    t.observed_boxes.clear()
    t.observed_scores.clear()
    for i in range(n):
        t.record(start + i, box + np.array([i, i, i, i], dtype=np.float64), 0.5 + i * 0.01)
    return t


def border_strip(frame: np.ndarray, *, frame_index: int) -> np.ndarray:
    """Peak colour of the drawn box's left edge on a given frame.

    `track()` puts observation i at [100+i, 100+i, 120+i, 120+i] and the box is drawn 2px
    outside that, so the left edge sits at x = 98 + i. Sampling the edge rather than a
    bounding region matters: the captions are white, and a region containing any text
    saturates every channel and would pass whatever colour the box actually is.
    """
    left = 98 + frame_index
    strip = frame[110 + frame_index : 118 + frame_index, left - 1 : left + 2]
    return np.asarray(strip, dtype=np.int32).reshape(-1, 3).max(axis=0)


class TestTheVisualCannotOutclaimTheRecord:
    """CLAUDE.md 1.3, applied to pixels rather than to prose."""

    @pytest.mark.parametrize("reason", sorted(UNCERTAIN_REASONS))
    def test_every_uncertainty_has_an_on_screen_caption(self, reason: Uncertainty) -> None:
        """Parametrised over the enum, so a new reason fails here rather than shipping a
        red box with nothing beside it."""
        overlay = TrackOverlay(
            track_number=1,
            track_id=f"{JOB}-t1",
            verdict=Verdict.ESCALATE,
            rule_id="uncertain.no_classifier",
            uncertainty=reason,
        )
        caveat = overlay.caveat
        assert caveat is not None
        assert "NOT CLASSIFIED" in caveat

    def test_a_confident_verdict_carries_no_caveat(self) -> None:
        """The caption is a correction. Printing it on a decided verdict would be its own
        false claim - that the system did not classify when it did."""
        overlay = TrackOverlay(
            track_number=1,
            track_id=f"{JOB}-t1",
            verdict=Verdict.ESCALATE,
            rule_id="escalate.sustained_hover",
            uncertainty=Uncertainty.NONE,
        )
        assert overlay.caveat is None

    def test_an_uncaptioned_uncertainty_raises_rather_than_drawing_a_bare_box(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tayr.render.annotate.UNCERTAINTY_CAPTIONS", {})
        with pytest.raises(ConfigError, match="no on-screen caption"):
            _caveat_for(Uncertainty.NO_CLASSIFIER_TRAINED)

    def test_captions_cover_the_enum_and_nothing_else(self) -> None:
        assert set(UNCERTAINTY_CAPTIONS) == set(UNCERTAIN_REASONS)
        assert Uncertainty.NONE not in UNCERTAINTY_CAPTIONS

    def test_every_verdict_has_a_colour(self) -> None:
        assert set(VERDICT_COLOURS) == set(Verdict)


class TestFormingTracksAreNotColouredAsDecided:
    """A track below `min_hits` had no verdict at that frame, and the colour says so."""

    def test_the_first_observations_are_grey(self) -> None:
        overlays = build_overlays([decision(1, Verdict.ESCALATE)], job_id=JOB)
        by_frame = _boxes_by_frame([track(1, n=6)], overlays, min_hits=3)

        assert [by_frame[f][0].forming for f in range(6)] == [
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        assert by_frame[0][0].colour == FORMING_COLOUR
        assert by_frame[2][0].colour == VERDICT_COLOURS[Verdict.ESCALATE]

    def test_the_boundary_matches_the_trackers_own_promotion_rule(self) -> None:
        """`_spawn` records observation 0 with hits=1 and `_apply` promotes at
        `hits >= min_hits`, so observation `min_hits - 1` is the first confirmed one.
        Off by one here would colour an unconfirmed box as a verdict."""
        for min_hits in (1, 2, 3, 5):
            by_frame = _boxes_by_frame(
                [track(1, n=6)],
                build_overlays([decision(1, Verdict.WATCH)], job_id=JOB),
                min_hits=min_hits,
            )
            forming = [by_frame[f][0].forming for f in range(6)]
            assert forming.count(True) == min_hits - 1, f"min_hits={min_hits}"

    def test_a_forming_box_says_forming_rather_than_naming_a_verdict(self) -> None:
        by_frame = _boxes_by_frame(
            [track(1)], build_overlays([decision(1, Verdict.ESCALATE)], job_id=JOB), min_hits=3
        )
        assert "forming" in by_frame[0][0].label
        assert "ESCALATE" not in by_frame[0][0].label
        assert "ESCALATE" in by_frame[3][0].label


class TestTracksWithNoDecision:
    def test_they_are_drawn_undecided_not_dropped_and_not_coloured(self) -> None:
        """Silently omitting them would hide detections the run really made; colouring
        them would invent a verdict that was never computed."""
        by_frame = _boxes_by_frame([track(7)], overlays={}, min_hits=3)
        box = by_frame[4][0]
        assert box.overlay is None
        assert box.colour == FORMING_COLOUR
        assert "undecided" in box.label
        assert "t7" in box.label


class TestOverlayLookup:
    def test_a_track_id_from_another_job_raises(self) -> None:
        """Dropping it would draw a real verdict as undecided, quietly."""
        stray = AgentDecision(
            track_id="other-job-t1",
            job_id="other-job",
            site_id="demo-north",
            decision=VerdictDecision(
                Verdict.DISMISS, Attention.ROUTINE, Uncertainty.NONE, "dismiss.test"
            ),
        )
        with pytest.raises(ConfigError, match="does not belong to job"):
            build_overlays([stray], job_id=JOB)

    def test_an_unparseable_suffix_raises(self) -> None:
        stray = AgentDecision(
            track_id=f"{JOB}-tabc",
            job_id=JOB,
            site_id="demo-north",
            decision=VerdictDecision(
                Verdict.DISMISS, Attention.ROUTINE, Uncertainty.NONE, "dismiss.test"
            ),
        )
        with pytest.raises(ConfigError, match="cannot recover a tracker id"):
            build_overlays([stray], job_id=JOB)

    def test_the_verdict_is_copied_from_the_record_not_recomputed(self) -> None:
        record = decision(
            3,
            Verdict.ESCALATE,
            uncertainty=Uncertainty.NO_CLASSIFIER_TRAINED,
            rule_id="uncertain.no_classifier",
        )
        overlay = build_overlays([record], job_id=JOB)[3]
        assert overlay.verdict is record.decision.verdict
        assert overlay.rule_id == record.decision.rule_id
        assert overlay.uncertainty is record.decision.uncertainty


@requires_cv_extra
class TestRenderEndToEnd:
    def scene(self, path: Path, *, frames: int = 12, fps: int = 25) -> Path:
        import av

        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(frames):
            frame = np.full((240, 320, 3), 30, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return path

    def render(self, tmp_path: Path, verdict: Verdict = Verdict.ESCALATE) -> object:
        source = self.scene(tmp_path / "scene.mp4")
        return render_annotated_video(
            source,
            tmp_path / "annotated.mp4",
            tracks=[track(1, n=10)],
            decisions=[decision(1, verdict)],
            job_id=JOB,
            tracker_config=TrackerConfig(min_hits=3),
        )

    def test_it_writes_a_decodable_video_of_the_same_length(self, tmp_path: Path) -> None:
        result = self.render(tmp_path)
        assert result.path.is_file()  # type: ignore[attr-defined]
        assert result.frames_written == 12  # type: ignore[attr-defined]
        assert result.boxes_drawn == 10  # type: ignore[attr-defined]

        from tayr.worker.pipeline import iter_frames

        decoded = list(iter_frames(result.path))  # type: ignore[attr-defined]
        assert len(decoded) == 12
        assert decoded[0].shape == (240, 320, 3)

    @pytest.mark.parametrize(
        ("verdict", "channel"),
        [(Verdict.ESCALATE, 0), (Verdict.DISMISS, 1)],
    )
    def test_the_box_is_actually_the_verdicts_colour(
        self, tmp_path: Path, verdict: Verdict, channel: int
    ) -> None:
        """Decodes the output and looks at the pixels. Catches the failure a unit test on
        the colour table cannot: frames are RGB here and BGR in `datasets/preview.py`, and
        a swap would silently turn every escalation blue."""
        result = self.render(tmp_path, verdict)

        from tayr.worker.pipeline import iter_frames

        # Frame 5: past min_hits, so the box carries its verdict colour rather than grey.
        frame = list(iter_frames(result.path))[5]  # type: ignore[attr-defined]
        edge = border_strip(frame, frame_index=5)
        dominant = int(np.argmax(edge))
        assert dominant == channel, f"{verdict.value} border is {edge}, not channel {channel}"

    def test_a_forming_box_is_grey_on_screen_not_just_in_the_data(self, tmp_path: Path) -> None:
        """The pixels are the claim. A box drawn red before its track was confirmed shows
        a verdict that did not exist yet, whatever the dataclass says."""
        source = self.scene(tmp_path / "scene.mp4")
        render_annotated_video(
            source,
            tmp_path / "annotated.mp4",
            tracks=[track(1, n=10)],
            decisions=[decision(1, Verdict.ESCALATE)],
            job_id=JOB,
            tracker_config=TrackerConfig(min_hits=6),
        )

        from tayr.worker.pipeline import iter_frames

        frames = list(iter_frames(tmp_path / "annotated.mp4"))
        forming = border_strip(frames[1], frame_index=1)
        decided = border_strip(frames[8], frame_index=8)

        assert int(forming.max() - forming.min()) < 30, f"forming border is not neutral: {forming}"
        assert int(decided[0]) > int(decided[1]) + 30, f"decided border is not red: {decided}"

    def test_a_run_note_says_how_many_escalations_were_uncertainties(self, tmp_path: Path) -> None:
        source = self.scene(tmp_path / "scene.mp4")
        result = render_annotated_video(
            source,
            tmp_path / "annotated.mp4",
            tracks=[track(1, n=10), track(2, n=10)],
            decisions=[
                decision(
                    1,
                    Verdict.ESCALATE,
                    uncertainty=Uncertainty.NO_CLASSIFIER_TRAINED,
                    rule_id="uncertain.no_classifier",
                ),
                decision(2, Verdict.ESCALATE, rule_id="escalate.sustained_hover"),
            ],
            job_id=JOB,
        )
        assert any("1 of 2 escalation(s) came from an uncertainty" in n for n in result.notes)

    def test_a_track_without_a_decision_is_reported_not_hidden(self, tmp_path: Path) -> None:
        source = self.scene(tmp_path / "scene.mp4")
        result = render_annotated_video(
            source,
            tmp_path / "annotated.mp4",
            tracks=[track(9, n=8)],
            decisions=[],
            job_id=JOB,
        )
        assert any("no decision record" in note for note in result.notes)
        assert result.boxes_drawn == 8

    def test_the_limits_are_enforced_on_the_second_decode_too(self, tmp_path: Path) -> None:
        """They guard a decode, not a file, and this is a decode."""
        from tayr.security.uploads import UploadRejectedError, VideoLimits

        source = self.scene(tmp_path / "scene.mp4")
        with pytest.raises(UploadRejectedError):
            render_annotated_video(
                source,
                tmp_path / "annotated.mp4",
                tracks=[],
                decisions=[],
                job_id=JOB,
                limits=VideoLimits(max_width=64, max_height=64),
            )


@requires_cv_extra
class TestRenderingIsOptional:
    def test_no_video_is_written_unless_asked(self, tmp_path: Path) -> None:
        """The headless path pays nothing for a file it will not open."""
        import av

        from tayr.agent.live import run_watch
        from tayr.config import Config

        path = tmp_path / "clip.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for _ in range(10):
            frame = np.full((120, 160, 3), 20, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        run = run_watch(
            path,
            Config.model_validate({"device": "cpu"}),
            output_dir=tmp_path / "out",
            site_registry_path=Path(__file__).resolve().parents[1]
            / "configs"
            / "sites"
            / "demo.yaml",
            repo=Path(__file__).resolve().parents[1],
            run_id="no-render",
            allow_stub=True,
        )
        assert run.render is None
        assert not (tmp_path / "out" / "annotated.mp4").exists()
