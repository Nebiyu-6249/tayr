"""Draw a run's boxes and verdicts onto the video they came from.

## The rule this module exists to obey

**The picture may not claim more than the record does.** A red box is the most
persuasive object in this project - it is read as "the system found a drone" by everyone
who has ever seen a detection demo - and for every track in Tayr today that reading is
wrong. There is no trained classifier, so the verdict behind most red boxes is
`uncertain.no_classifier`: an escalation *because the system cannot tell*, which is the
opposite claim.

So every box carries its rule id, and a box whose verdict came from an uncertainty
carries the reason spelled out on screen (`NOT CLASSIFIED - no classifier trained`)
rather than only in JSON that nobody pauses the video to read. `TrackOverlay.caveat` is
where that lives, and `test_render.py` asserts it appears for every reason in
`UNCERTAIN_REASONS` - so adding an uncertainty without giving it a caption fails the
suite rather than shipping a red box with nothing next to it.

The same rule covers the corner overlay: it counts verdicts, and separately counts how
many of the escalations were uncertainties. "6 ESCALATE" alone would overstate the run.

## What is drawn

One box per *observed* detection: a box appears on a frame only where a detection was
actually associated to that track on that frame. Gaps stay empty rather than being
filled with the Kalman filter's prediction - the filter's guess is not an observation,
and drawing it would show the video a smoothness that was never measured.

Colour is the track's verdict, except during the first `min_hits - 1` observations, when
the track was still TENTATIVE and had no verdict to show. Those are grey. The colour
therefore changes mid-track exactly where the tracker confirmed it, which is honest
about when the system actually knew anything.

## Why a second decode

Verdicts are known only after the last frame - a track's motion features need its whole
history - so nothing can be drawn during the first pass. The alternative, holding every
decoded frame in memory until the verdicts arrive, is the exact failure the streaming
decode exists to prevent: a 300-second 4K sequence is hundreds of gigabytes decoded.
Rendering therefore decodes a second time, and re-probes first, because the limits are
enforced before a decode rather than once per file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tayr.agent.records import AgentDecision
from tayr.agent.verdicts import UNCERTAIN_REASONS, Uncertainty, Verdict
from tayr.config import TrackerConfig
from tayr.errors import ConfigError, DependencyUnavailableError
from tayr.geometry import pixels_on_target
from tayr.security.uploads import MediaProperties, VideoLimits
from tayr.tracking.tracker import Track
from tayr.worker.pipeline import iter_frames
from tayr.worker.probe import probe_video

#: RGB, not BGR. Frames arrive from PyAV as rgb24 and are handed back the same way, so
#: the tuples here are read in the order they are written - unlike `datasets/preview.py`,
#: which works on `cv2.imread` output and is therefore BGR. Getting this backwards
#: swaps red and blue, which on a verdict overlay swaps ESCALATE for nothing at all.
VERDICT_COLOURS: dict[Verdict, tuple[int, int, int]] = {
    Verdict.ESCALATE: (235, 64, 52),
    Verdict.WATCH: (240, 173, 32),
    Verdict.DISMISS: (60, 200, 90),
}

#: A track below `min_hits` has no verdict yet. Grey says "not decided", where any
#: verdict colour would say something the record does not.
FORMING_COLOUR: tuple[int, int, int] = (150, 150, 150)

TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)
PANEL_COLOUR: tuple[int, int, int] = (18, 18, 18)
CAVEAT_COLOUR: tuple[int, int, int] = (255, 214, 102)

#: Written next to any box whose ESCALATE came from not knowing rather than from
#: knowing. Every member of `UNCERTAIN_REASONS` needs an entry; `_caveat_for` raises if
#: one is missing, so a new reason cannot ship as a bare red box.
UNCERTAINTY_CAPTIONS: dict[Uncertainty, str] = {
    Uncertainty.NO_CLASSIFIER_TRAINED: "NOT CLASSIFIED - no classifier trained",
    Uncertainty.CLASSIFIER_UNDETERMINED: "NOT CLASSIFIED - classifier undetermined",
    Uncertainty.TRACK_TOO_SHORT: "NOT CLASSIFIED - track too short to judge",
    Uncertainty.TOOL_FAILURE: "NOT CLASSIFIED - a tool call failed",
    Uncertainty.ROUND_CAP_REACHED: "NOT CLASSIFIED - reasoning hit its round cap",
}

#: Codec and pixel format for the output. mpeg4 in an mp4 container plays everywhere
#: without a licensing question, and is what the demo scene encoder already uses.
OUTPUT_CODEC = "mpeg4"
OUTPUT_PIX_FMT = "yuv420p"


@dataclass(frozen=True, slots=True)
class TrackOverlay:
    """What the decision record says about one track, in drawable form.

    Built from an `AgentDecision`, never from a detection: the box on screen and the
    verdict beside it must come from the same source of truth as `decisions.json`.
    """

    track_number: int
    track_id: str
    verdict: Verdict
    rule_id: str
    uncertainty: Uncertainty

    @property
    def colour(self) -> tuple[int, int, int]:
        return VERDICT_COLOURS[self.verdict]

    @property
    def caveat(self) -> str | None:
        """The on-screen correction for a verdict that came from not knowing."""
        return _caveat_for(self.uncertainty)


@dataclass(frozen=True, slots=True)
class _Box:
    """One observation, ready to draw."""

    track_number: int
    xyxy: npt.NDArray[np.float64]
    score: float
    pixels_on_target: float
    forming: bool
    overlay: TrackOverlay | None
    """None when no decision was recorded for the track - drawn grey and labelled
    `undecided`, not silently skipped and not coloured as though a verdict existed."""

    @property
    def decided(self) -> bool:
        """A box shows a verdict only where the record has one to show."""
        return self.overlay is not None and not self.forming

    @property
    def colour(self) -> tuple[int, int, int]:
        return self.overlay.colour if self.decided and self.overlay else FORMING_COLOUR

    @property
    def label(self) -> str:
        if self.forming:
            state = "forming"
        elif self.overlay is None:
            state = "undecided"
        else:
            state = self.overlay.verdict.value.upper()
        return f"t{self.track_number} {state} {self.pixels_on_target:.0f}px conf {self.score:.2f}"


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What the render produced, and what it had to leave out."""

    path: Path
    frames_written: int
    boxes_drawn: int
    media: MediaProperties
    notes: list[str] = field(default_factory=list)


def _caveat_for(uncertainty: Uncertainty) -> str | None:
    if uncertainty is Uncertainty.NONE:
        return None
    caption = UNCERTAINTY_CAPTIONS.get(uncertainty)
    if caption is None:
        # Loud rather than silent: a red box with no caption is the failure this module
        # exists to prevent, and a new Uncertainty member is exactly how it would arrive.
        raise ConfigError(
            f"uncertainty {uncertainty.value!r} has no on-screen caption in "
            "UNCERTAINTY_CAPTIONS. An escalation from uncertainty must say so on the "
            "video, not only in the JSON."
        )
    return caption


def build_overlays(decisions: Sequence[AgentDecision], *, job_id: str) -> dict[int, TrackOverlay]:
    """Index decisions by tracker id, the number the boxes are keyed on.

    `track_id` is `f"{job_id}-t{n}"`, built in `agent.live`. Parsing it back is
    unpleasant but keeps `AgentDecision` free of a tracker-internal integer, and the
    suffix is checked rather than assumed - a mismatch raises instead of dropping a
    verdict and drawing the box grey.
    """
    overlays: dict[int, TrackOverlay] = {}
    prefix = f"{job_id}-t"
    for record in decisions:
        if not record.track_id.startswith(prefix):
            raise ConfigError(
                f"decision track_id {record.track_id!r} does not belong to job {job_id!r}. "
                "Rendering would silently drop its verdict and draw the box as undecided."
            )
        suffix = record.track_id.removeprefix(prefix)
        if not suffix.isdigit():
            raise ConfigError(f"cannot recover a tracker id from track_id {record.track_id!r}")
        decision = record.decision
        overlays[int(suffix)] = TrackOverlay(
            track_number=int(suffix),
            track_id=record.track_id,
            verdict=decision.verdict,
            rule_id=decision.rule_id,
            uncertainty=decision.uncertainty,
        )
    return overlays


def _boxes_by_frame(
    tracks: Sequence[Track],
    overlays: dict[int, TrackOverlay],
    *,
    min_hits: int,
) -> dict[int, list[_Box]]:
    """Lay every track's observations out per frame.

    A track's k-th observation was recorded when it had k+1 hits, and `ByteTracker`
    promotes at `hits >= min_hits` - so observations before index `min_hits - 1` were
    made while the track was still TENTATIVE and had no verdict behind them.
    """
    by_frame: dict[int, list[_Box]] = {}
    for track in tracks:
        overlay = overlays.get(track.track_id)
        for index, (frame_index, box, score) in enumerate(
            zip(track.observed_frames, track.observed_boxes, track.observed_scores, strict=True)
        ):
            arr = np.asarray(box, dtype=np.float64).reshape(4)
            by_frame.setdefault(frame_index, []).append(
                _Box(
                    track_number=track.track_id,
                    xyxy=arr,
                    score=float(score),
                    pixels_on_target=float(pixels_on_target(arr)),
                    forming=index < max(0, min_hits - 1),
                    overlay=overlay,
                )
            )
    return by_frame


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    *,
    tracks: Sequence[Track],
    decisions: Sequence[AgentDecision],
    job_id: str,
    tracker_config: TrackerConfig | None = None,
    limits: VideoLimits | None = None,
    max_frames: int | None = None,
) -> RenderResult:
    """Re-decode `video_path`, draw the run onto it, and encode to `output_path`.

    The limits are enforced again here. They guard a decode, not a file, and this is a
    second decode - re-probing costs a header read and removes the possibility that a
    render path grows its own way past them.
    """
    cv2 = _require_cv2()
    av = _require_av()

    media = probe_video(video_path, limits=limits)
    config = tracker_config or TrackerConfig()
    overlays = build_overlays(decisions, job_id=job_id)
    by_frame = _boxes_by_frame(tracks, overlays, min_hits=config.min_hits)

    undecided = sorted({t.track_id for t in tracks} - set(overlays))
    notes: list[str] = []
    if undecided:
        notes.append(
            f"{len(undecided)} track(s) have no decision record and are drawn grey: "
            f"{undecided}. Grey means undecided, and that is what these are."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = media.fps if media.fps > 0 else 30.0
    counts: dict[Verdict, int] = dict.fromkeys(Verdict, 0)
    for overlay in overlays.values():
        counts[overlay.verdict] += 1
    uncertain_escalations = sum(
        1
        for overlay in overlays.values()
        if overlay.verdict is Verdict.ESCALATE and overlay.uncertainty in UNCERTAIN_REASONS
    )

    container = av.open(str(output_path), mode="w")
    frames_written = 0
    boxes_drawn = 0
    try:
        stream = container.add_stream(OUTPUT_CODEC, rate=round(fps))
        stream.width, stream.height = media.width, media.height
        stream.pix_fmt = OUTPUT_PIX_FMT

        for frame_index, frame in enumerate(iter_frames(video_path, max_frames=max_frames)):
            canvas = np.ascontiguousarray(frame)
            drawn = _draw_boxes(cv2, canvas, by_frame.get(frame_index, ()))
            _draw_panel(
                cv2,
                canvas,
                frame_index=frame_index,
                elapsed=frame_index / fps,
                counts=counts,
                uncertain_escalations=uncertain_escalations,
            )
            boxes_drawn += drawn
            container.mux(stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")))
            frames_written += 1

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    if uncertain_escalations:
        notes.append(
            f"{uncertain_escalations} of {counts[Verdict.ESCALATE]} escalation(s) came from "
            "an uncertainty, not from a finding. Each is captioned on the video as well as "
            "counted in the corner, because a red box reads as a positive identification "
            "and these are the opposite of one."
        )

    return RenderResult(
        path=output_path,
        frames_written=frames_written,
        boxes_drawn=boxes_drawn,
        media=media,
        notes=notes,
    )


#: Vertical pitch between stacked caption lines, in pixels. Matches the 0.4-scale
#: FONT_HERSHEY_SIMPLEX cap height with a little air.
LINE_HEIGHT = 12

#: Two captions closer than this horizontally are treated as overlapping. Generous,
#: because a caption is far wider than the box it belongs to.
LABEL_WIDTH = 220


class _Captions:
    """Places caption lines so that overlapping boxes do not overprint each other.

    Small targets clustered in one part of the sky is the normal case here, not the
    exception - which means several boxes want the same few rows of pixels for their
    captions, and unplaced text turns into an unreadable smear exactly where the
    interesting thing is happening. Each line is pushed down until it finds a free row.
    """

    __slots__ = ("_height", "_used")

    def __init__(self, height: int) -> None:
        self._height = height
        self._used: list[tuple[int, int]] = []

    def place(self, x: int, y: int) -> int:
        """A free baseline at or below `y` for a caption starting at `x`."""
        row = max(LINE_HEIGHT, min(self._height - 4, y))
        while any(
            abs(row - used_y) < LINE_HEIGHT and abs(x - used_x) < LABEL_WIDTH
            for used_x, used_y in self._used
        ):
            row += LINE_HEIGHT
            if row > self._height - 4:
                # Out of frame. Better an overlap at the bottom edge than a caption
                # written outside the canvas and silently lost.
                row = self._height - 4
                break
        self._used.append((x, row))
        return row


def _draw_boxes(cv2: Any, canvas: npt.NDArray[np.uint8], boxes: Sequence[_Box]) -> int:
    """Draw one frame's boxes. Returns how many were drawn."""
    height = canvas.shape[0]
    captions = _Captions(height)
    # Top-down, so the stacking order on screen matches the order of the boxes down the
    # frame rather than the order tracks happen to have been created in.
    for item in sorted(boxes, key=lambda b: float(b.xyxy[1])):
        x1, y1, x2, y2 = (round(float(v)) for v in item.xyxy)
        colour = item.colour
        left = max(0, x1 - 2)
        # Outside the box, never over it: a 1px border on a 10px target would cover a
        # fifth of the pixels the box is about, and small targets are the whole subject.
        cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 1)
        # White text, coloured box. The verdict colour has far less luminance contrast
        # against sky than white does, and at this size it is the first thing lossy
        # encoding destroys - so colour carries the verdict on the box, where a shape
        # survives compression, and the caption says the same word in readable type.
        _text(cv2, canvas, item.label, (left, captions.place(left, y1 - 6)))

        if not item.decided or item.overlay is None:
            continue

        # Under the box rather than over it, so a caption cannot be cropped off the top
        # of the frame - and the caveat goes first, because it is the line that stops a
        # red box being read as an identification.
        caveat = item.overlay.caveat
        if caveat:
            _text(cv2, canvas, caveat, (left, captions.place(left, y2 + 14)), CAVEAT_COLOUR)
        _text(
            cv2,
            canvas,
            item.overlay.rule_id,
            (left, captions.place(left, y2 + 14 + (LINE_HEIGHT if caveat else 0))),
        )
    return len(boxes)


def _draw_panel(
    cv2: Any,
    canvas: npt.NDArray[np.uint8],
    *,
    frame_index: int,
    elapsed: float,
    counts: dict[Verdict, int],
    uncertain_escalations: int,
) -> None:
    """Corner overlay: where we are, and what the run decided so far.

    The counts are the run's totals rather than a running tally. A per-frame tally would
    imply the agent decides as the video plays; it does not - every verdict is computed
    after the last frame, from the whole track.
    """
    width = canvas.shape[1]
    panel_width = min(width, 340)
    cv2.rectangle(canvas, (0, 0), (panel_width, 62), PANEL_COLOUR, -1)

    minutes, seconds = divmod(elapsed, 60.0)
    _text(cv2, canvas, f"frame {frame_index}   {int(minutes):02d}:{seconds:05.2f}", (6, 16))

    tally = "  ".join(f"{v.value.upper()} {counts[v]}" for v in Verdict)
    _text(cv2, canvas, tally, (6, 34))

    if uncertain_escalations:
        _text(
            cv2,
            canvas,
            f"{uncertain_escalations} escalation(s) = uncertainty, not a finding",
            (6, 52),
            CAVEAT_COLOUR,
        )
    else:
        _text(cv2, canvas, "verdicts computed after the last frame", (6, 52), FORMING_COLOUR)


def _text(
    cv2: Any,
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int] = TEXT_COLOUR,
) -> None:
    """Draw text with a dark outline, so it stays readable against bright sky."""
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_COLOUR, 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise DependencyUnavailableError(
            "OpenCV is required to draw the annotated video but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'"
        ) from exc
    return cv2


def _require_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise DependencyUnavailableError(
            "PyAV is required to encode the annotated video but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'"
        ) from exc
    return av
