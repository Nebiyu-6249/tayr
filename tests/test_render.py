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
from tayr.config import RenderConfig, TrackerConfig
from tayr.errors import ConfigError
from tayr.render.annotate import (
    CODEC_FALLBACKS,
    CRF_CAPABLE,
    FORMING_COLOUR,
    UNCERTAINTY_CAPTIONS,
    VERDICT_COLOURS,
    TrackOverlay,
    _boxes_by_frame,
    _caveat_for,
    _even,
    build_overlays,
    render_annotated_video,
    resolve_codec,
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


class TestCodecResolution:
    """The encode is the demo artefact. Which encoder ran must never be a guess."""

    def test_an_available_encoder_is_used_without_comment(self) -> None:
        codec, warning = resolve_codec("mpeg4")
        assert codec == "mpeg4"
        assert warning is None

    @requires_cv_extra
    def test_h264_is_available_in_this_build(self) -> None:
        """Not an assumption about FFmpeg: the default has to actually resolve, or every
        render silently falls back and the file is soft again."""
        codec, warning = resolve_codec("h264")
        assert codec == "h264"
        assert warning is None

    def test_an_absent_encoder_falls_back_and_says_so(self) -> None:
        """Silence here is the failure mode: a soft file with no explanation."""
        codec, warning = resolve_codec("libtotallyfictional")
        assert codec in CODEC_FALLBACKS
        assert warning is not None
        assert "CODEC FALLBACK" in warning
        assert "libtotallyfictional" in warning

    def test_encoders_are_probed_for_writing_not_merely_listed(self) -> None:
        """`av.codecs_available` lists decoders too. A build that can read H.264 and not
        write it would pass a membership check and fail at the first encode() call."""
        import av

        from tayr.render import annotate

        assert "libtotallyfictional" not in av.codecs_available
        # A decode-only name would be the real hazard; assert the probe uses Codec(_, "w")
        # rather than the list, by checking the source it actually calls.
        assert 'Codec(name, "w")' in Path(annotate.__file__).read_text(encoding="utf-8")


class TestEncodeSettingsAreReported:
    def test_mpeg4_reports_that_crf_did_not_apply(self) -> None:
        """FFmpeg ignores crf on mpeg4 silently. Reporting it as applied would be a lie
        about why the file looks the way it does."""
        assert "mpeg4" not in CRF_CAPABLE

    def test_even_dimensions_are_forced(self) -> None:
        """H.264 with 4:2:0 chroma cannot represent an odd width; libx264 rejects the
        stream rather than rounding."""
        assert _even(1081) == 1080
        assert _even(1080) == 1080
        assert _even(1.0) == 2, "never below 2"


@requires_cv_extra
class TestEncodeOptions:
    def scene(self, path: Path, *, frames: int = 12, fps: int = 25) -> Path:
        import av

        rng = np.random.default_rng(3)
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(frames):
            # Noise, so the encode has something to spend bits on and CRF can differ.
            frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return path

    def render(self, tmp_path: Path, name: str, **kwargs: object) -> object:
        source = self.scene(tmp_path / "scene.mp4")
        return render_annotated_video(
            source,
            tmp_path / f"{name}.mp4",
            tracks=[track(1, n=10)],
            decisions=[decision(1, Verdict.ESCALATE)],
            job_id=JOB,
            render_config=RenderConfig(**kwargs),  # type: ignore[arg-type]
        )

    def test_the_default_is_h264_with_crf(self, tmp_path: Path) -> None:
        result = self.render(tmp_path, "default")
        assert result.codec in ("h264", "libx264")  # type: ignore[attr-defined]
        assert result.crf == 18  # type: ignore[attr-defined]

    def test_a_lower_crf_produces_a_larger_file(self, tmp_path: Path) -> None:
        """CRF is quality-targeted, so this is the direction that proves it applied at
        all - an ignored option would give two identical files."""
        good = self.render(tmp_path, "crf14", crf=14).size_bytes  # type: ignore[attr-defined]
        poor = self.render(tmp_path, "crf40", crf=40).size_bytes  # type: ignore[attr-defined]
        assert good > poor, f"crf 14 gave {good} bytes, crf 40 gave {poor}"

    def test_crf_is_reported_as_not_applied_on_an_encoder_without_it(self, tmp_path: Path) -> None:
        result = self.render(tmp_path, "mpeg", codec="mpeg4", crf=18)
        assert result.crf is None  # type: ignore[attr-defined]
        assert any("CRF NOT APPLIED" in n for n in result.notes)  # type: ignore[attr-defined]
        assert "no CRF" in result.settings_line()  # type: ignore[attr-defined]

    def test_scale_shrinks_the_output_and_says_so(self, tmp_path: Path) -> None:
        result = self.render(tmp_path, "half", scale=0.5)
        assert (result.width, result.height) == (160, 120)  # type: ignore[attr-defined]
        assert any("SCALED" in n for n in result.notes)  # type: ignore[attr-defined]

        from tayr.worker.pipeline import iter_frames

        first = next(iter(iter_frames(result.path)))  # type: ignore[attr-defined]
        assert first.shape == (120, 160, 3)

    def test_captions_keep_their_size_when_the_video_is_scaled(self, tmp_path: Path) -> None:
        """Drawing then shrinking would scale the text down with the picture, which is
        backwards: a smaller file whose captions are unreadable is worth nothing."""
        full = self.render(tmp_path, "full")
        half = self.render(tmp_path, "half2", scale=0.5)

        from tayr.worker.pipeline import iter_frames

        def caption_rows(path: Path, width: int) -> int:
            """Rows of the corner panel's text block that carry bright pixels."""
            frame = list(iter_frames(path))[5][:62, : min(width, 340)]
            return int((frame.max(axis=(1, 2)) > 200).sum())

        assert caption_rows(half.path, 160) >= caption_rows(full.path, 320) * 0.6  # type: ignore[attr-defined]

    def test_a_caption_near_the_right_edge_is_shifted_in_not_clipped(self, tmp_path: Path) -> None:
        """Text running off the edge is text the viewer silently does not get, and the
        longest line is the NOT CLASSIFIED caveat - exactly the one that matters."""
        # A flat scene, not the noisy one: this test reads pixel brightness at the frame
        # edge, and random noise would supply bright pixels of its own and pass or fail
        # for reasons that have nothing to do with where the caption was drawn.
        import av

        source = tmp_path / "flat.mp4"
        container = av.open(str(source), mode="w")
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(12):
            flat = np.full((240, 320, 3), 24, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(flat, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        edge = track(1, n=10)
        # Push the track hard against the right edge of a 320px frame.
        edge.observed_boxes = [
            box + np.array([200.0, 0.0, 200.0, 0.0]) for box in edge.observed_boxes
        ]
        result = render_annotated_video(
            source,
            tmp_path / "edge.mp4",
            tracks=[edge],
            decisions=[
                decision(
                    1,
                    Verdict.ESCALATE,
                    uncertainty=Uncertainty.NO_CLASSIFIER_TRAINED,
                    rule_id="uncertain.no_classifier",
                )
            ],
            job_id=JOB,
        )

        from tayr.worker.pipeline import iter_frames

        frame = list(iter_frames(result.path))[5]
        assert int((frame[:, -1].max(axis=1) > 200).sum()) == 0, "a glyph reaches the edge"
        assert not any("CAPTIONS DO NOT FIT" in n for n in result.notes)

    def test_a_frame_narrower_than_its_captions_says_so(self, tmp_path: Path) -> None:
        """Past a point no placement helps - the frame is narrower than the text itself.
        Clipping is information loss, so it is reported rather than hidden."""
        result = self.render(tmp_path, "tiny", scale=0.25)
        assert any("CAPTIONS DO NOT FIT" in n for n in result.notes)  # type: ignore[attr-defined]
        assert any("--render-scale" in n for n in result.notes)  # type: ignore[attr-defined]

    def test_the_settings_line_names_codec_resolution_and_size(self, tmp_path: Path) -> None:
        line = self.render(tmp_path, "line").settings_line()  # type: ignore[attr-defined]
        assert "320x240" in line
        assert "CRF 18" in line
        assert "MB" in line
        assert "kbps" in line


@requires_cv_extra
class TestWhyTheDefaultIsH264:
    """Evidence for the default, not a preference.

    The first renderer wrote mpeg4 and produced a visibly soft file: at 1080p the box
    outlines smeared, which defeats an overlay whose whole purpose is that a viewer can
    check the claim against the pixels. Measured on the content that matters - thin
    lines and small text on flat sky - rather than asserted.
    """

    def frames(self, n: int = 24) -> list[np.ndarray]:
        """Clean sky plus the overlay. Deliberately no injected noise: noise is
        incompressible and its error would swamp what the codecs do to lines and text,
        which is the thing being compared."""
        import cv2

        out = []
        for i in range(n):
            img = np.full((720, 1280, 3), 30, np.uint8)
            img[:240] = 52
            for k in range(4):
                x = 200 + i * 3 + k * 220
                cv2.rectangle(img, (x, 300 + k * 40), (x + 16, 316 + k * 40), (235, 64, 52), 1)
                cv2.putText(
                    img,
                    "t6 ESCALATE 13px conf 0.42",
                    (x - 40, 294 + k * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            out.append(img)
        return out

    def encode(self, path: Path, codec: str, crf: int | None) -> tuple[float, int]:
        """Round trip the frames and return (mean absolute RGB error, file size)."""
        from typing import cast

        import av

        source = self.frames()
        container = av.open(str(path), mode="w")
        # `add_stream` overloads on Literal codec names, so passing a str variable widens
        # the return to include audio and subtitle streams. Elsewhere in this file the
        # codec is a literal and mypy narrows it for free.
        stream = cast("av.video.stream.VideoStream", container.add_stream(codec, rate=25))
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        if crf is not None:
            stream.options = {"crf": str(crf)}
        for frame in source:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        decoder = av.open(str(path))
        decoded = [f.to_ndarray(format="rgb24") for f in decoder.decode(decoder.streams.video[0])]
        decoder.close()
        ref = np.asarray(source[-1], dtype=np.int32)
        got = np.asarray(decoded[-1], dtype=np.int32)
        return float(np.abs(ref - got).mean()), path.stat().st_size

    def test_the_default_beats_mpeg4_on_both_size_and_accuracy(self, tmp_path: Path) -> None:
        """Not a trade: H.264 at CRF 18 is smaller AND closer to the source.

        Comparing requested bitrates would measure nothing - the two encoders honour
        `bit_rate` very differently, and mpeg4 overshot a 100 kbps request by 4x in this
        build - so this compares what each actually produced.
        """
        x264_err, x264_size = self.encode(tmp_path / "x264.mp4", "libx264", 18)
        mpeg4_err, mpeg4_size = self.encode(tmp_path / "mpeg4.mp4", "mpeg4", None)

        assert x264_size < mpeg4_size, f"libx264 {x264_size} vs mpeg4 {mpeg4_size} bytes"
        assert x264_err < mpeg4_err, f"libx264 {x264_err:.3f} vs mpeg4 {mpeg4_err:.3f} error"
        # Margins measured at 6.6x on size and 23x on error, so a factor of two each way
        # is a loose floor that still fails if the default silently stops applying.
        assert x264_size * 2 < mpeg4_size
        assert x264_err * 2 < mpeg4_err
