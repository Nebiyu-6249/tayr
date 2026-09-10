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

## Encoding

This is the artefact people watch, so the encode is not cosmetic. The first version
wrote MPEG-4 Part 2 and produced a visibly soft file at 1080p, with box outlines smeared
into suggestions - which defeats an overlay whose purpose is that a viewer can check the
claim against the pixels.

H.264 at CRF 18 is not a trade against file size; it wins on both. Same 24 frames of
1280x720 sky with four annotated boxes, round-tripped and compared against the source
`[VERIFIED: 2026-09-10, PyAV 18.1.0]`:

    encoder            bytes   mean |err|
    mpeg4 (default)    86978        0.893
    libx264 CRF 18     13256        0.038
    libx264 CRF 28     10074        0.070
    libx264 CRF 40      8683        0.209

6.6x smaller and 23x more accurate; even CRF 40 is a tenth the size and still four times
closer than mpeg4. Requested *bitrates* are not comparable between the two - mpeg4
overshot a 100 kbps request by more than four times in this build - so size and error are
what get measured.

CRF is quality-targeted rather than bitrate-targeted, which suits content whose subject
is a 12px object against flat sky: still frames cost few bits and the busy ones get what
they need. Encoders without a CRF mode - mpeg4 among them - silently ignore the option,
so `RenderResult.crf` is None there and the run says the setting did not apply.

## Caption colour, measured rather than assumed

Captions are white and boxes are coloured. That was first justified as "lossy encoding
destroys thin coloured text", which blamed mpeg4 and predicted the constraint would lift
at H.264 CRF 18. **It does not**, and the reason it does not is worth writing down.

Round-tripping the same caption through each encoder and measuring the decoded glyph
pixels `[VERIFIED: 2026-09-10, mean absolute RGB error on glyph pixels / luminance
contrast against the decoded background]`:

    encoder   pix_fmt    colour   mean |err|   contrast
    mpeg4     yuv420p    white          24.9      148.8
    mpeg4     yuv420p    red            38.0       30.8
    libx264   yuv420p    white           1.8      169.2
    libx264   yuv420p    amber          18.1      138.0
    libx264   yuv420p    red            25.9       57.6
    libx264   yuv444p    red             3.2       53.3

Two separate effects, and H.264 fixes neither for red:

1. **Chroma subsampling, not the codec.** Red text is carried almost entirely in
   chroma, and `yuv420p` stores chroma at half resolution in each axis. The error only
   collapses at `yuv444p` (25.9 -> 3.2), which needs H.264 High 4:4:4 Predictive and is
   refused by many players - a bad trade for a file meant to be watched by other people.
2. **Luminance contrast is intrinsic to the colour.** Red (235, 64, 52) has a luminance
   of about 100 against white's 255, and the measured contrast barely moves between
   4:2:0 and 4:4:4 (57.6 vs 53.3). No encoder setting can recover that.

Amber at 138 is close enough to white's 169 to stay, which matters because the amber line
is the `NOT CLASSIFIED` caveat - the one caption that must be readable.

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
from tayr.config import RenderConfig, TrackerConfig
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

#: Pixel format for the output. 4:2:0 subsampling halves chroma resolution, which is
#: unkind to 1px coloured borders - but 4:4:4 needs H.264 High 4:4:4 Predictive, which
#: many players refuse. Compatibility wins for a file whose purpose is to be watched by
#: other people; the encoder's quality setting is where the legibility is bought back.
OUTPUT_PIX_FMT = "yuv420p"

#: Tried in order when the configured codec is unavailable. `mpeg4` is last because it
#: is MPEG-4 Part 2 - a much weaker codec than H.264 at the same bitrate, and the reason
#: the first version of this renderer produced a visibly soft file.
CODEC_FALLBACKS: tuple[str, ...] = ("libx264", "mpeg4")

#: Encoders that understand `crf`. Setting it on anything else is silently ignored by
#: FFmpeg, which is exactly the kind of quiet no-op that has to be reported instead.
CRF_CAPABLE: frozenset[str] = frozenset({"libx264", "h264", "libx265", "hevc", "libvpx-vp9"})


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
    """What the render produced, under what settings, and what it left out."""

    path: Path
    frames_written: int
    boxes_drawn: int
    media: MediaProperties
    codec: str = ""
    """The encoder actually used, which is not necessarily the one asked for."""

    crf: int | None = None
    """None when the encoder has no CRF and the setting did not apply."""

    width: int = 0
    height: int = 0
    scale: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.is_file() else 0

    @property
    def bitrate_kbps(self) -> float:
        """Mean bitrate over the encoded duration, in kbit/s."""
        seconds = self.frames_written / self.media.fps if self.media.fps > 0 else 0.0
        return self.size_bytes * 8 / seconds / 1000 if seconds > 0 else 0.0

    def settings_line(self) -> str:
        """One line naming what was encoded and how. Printed, never assumed."""
        quality = f"CRF {self.crf}" if self.crf is not None else "no CRF (encoder has none)"
        scale = "" if self.scale == 1.0 else f" @ {self.scale:g}x"
        return (
            f"{self.codec} {quality}  {self.width}x{self.height}{scale}  "
            f"{self.size_bytes / 1e6:.1f} MB  ~{self.bitrate_kbps:.0f} kbps"
        )


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


def resolve_codec(requested: str) -> tuple[str, str | None]:
    """The encoder to use, and a warning when it is not the one asked for.

    Asked in terms of encoders rather than `av.codecs_available`, which lists decoders
    too: a build can decode H.264 and be unable to produce it, and finding that out at
    the first `encode()` call means a half-written file and a stack trace.
    """
    from av.codec import Codec

    def usable(name: str) -> bool:
        try:
            Codec(name, "w")
        except ValueError:
            # UnknownCodecError subclasses ValueError. Anything else is not "absent".
            return False
        return True

    if usable(requested):
        return requested, None

    for candidate in CODEC_FALLBACKS:
        if candidate == requested or not usable(candidate):
            continue
        softer = (
            " That is MPEG-4 Part 2, a much weaker codec than H.264: expect a visibly "
            "softer picture, with thin box outlines and small captions smeared."
            if candidate == "mpeg4"
            else ""
        )
        return candidate, (
            f"CODEC FALLBACK: {requested!r} is not an encoder in this FFmpeg build, so "
            f"{candidate!r} was used instead.{softer} The boxes and verdicts are "
            "unaffected either way - only how well they survive the encode."
        )

    raise ConfigError(
        f"no usable video encoder. {requested!r} is absent and so is every fallback "
        f"({', '.join(CODEC_FALLBACKS)}). This FFmpeg build cannot write video."
    )


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    *,
    tracks: Sequence[Track],
    decisions: Sequence[AgentDecision],
    job_id: str,
    tracker_config: TrackerConfig | None = None,
    render_config: RenderConfig | None = None,
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
    encoding = render_config or RenderConfig()
    overlays = build_overlays(decisions, job_id=job_id)
    by_frame = _boxes_by_frame(tracks, overlays, min_hits=config.min_hits)

    undecided = sorted({t.track_id for t in tracks} - set(overlays))
    notes: list[str] = []
    if undecided:
        notes.append(
            f"{len(undecided)} track(s) have no decision record and are drawn grey: "
            f"{undecided}. Grey means undecided, and that is what these are."
        )

    codec, warning = resolve_codec(encoding.codec)
    if warning:
        notes.append(warning)

    crf: int | None = encoding.crf if codec in CRF_CAPABLE else None
    if crf is None:
        notes.append(
            f"CRF NOT APPLIED: {codec!r} has no constant-rate-factor mode, so "
            f"--render-crf {encoding.crf} had no effect on this file."
        )

    # Even dimensions: H.264 with 4:2:0 chroma cannot represent an odd width or height,
    # and libx264 rejects the stream rather than rounding.
    width, height = _even(media.width * encoding.scale), _even(media.height * encoding.scale)
    if (width, height) != (media.width, media.height):
        notes.append(
            f"SCALED {media.width}x{media.height} -> {width}x{height} "
            f"({encoding.scale:g}x). Frames are resized before the overlay is drawn, so "
            "captions keep their pixel size; boxes are scaled with the image."
        )

    too_narrow = _widest_caption(cv2, by_frame)
    if too_narrow > width:
        # Captions keep their pixel size when the video is scaled, which is what makes a
        # small file still readable - but past a point the frame is narrower than the
        # text itself and no placement helps. Clipping is information loss, so it is
        # reported rather than left for the viewer to not notice.
        notes.append(
            f"CAPTIONS DO NOT FIT: the widest is {too_narrow}px against a {width}px "
            f"frame, so some text is clipped. Raise --render-scale (currently "
            f"{encoding.scale:g}) if the captions matter more than the file size."
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
        stream = container.add_stream(codec, rate=round(fps))
        stream.width, stream.height = width, height
        stream.pix_fmt = OUTPUT_PIX_FMT
        if crf is not None:
            # Quality-targeted rather than bitrate-targeted: a still sky costs few bits
            # and a cluttered frame gets what it needs, which is the right trade for a
            # file whose subject is a 12px object.
            stream.options = {"crf": str(crf)}

        for frame_index, frame in enumerate(iter_frames(video_path, max_frames=max_frames)):
            canvas = _canvas(cv2, frame, width=width, height=height)
            drawn = _draw_boxes(cv2, canvas, by_frame.get(frame_index, ()), scale=encoding.scale)
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
        codec=codec,
        crf=crf,
        width=width,
        height=height,
        scale=encoding.scale,
        notes=notes,
    )


def _widest_caption(cv2: Any, by_frame: dict[int, list[_Box]]) -> int:
    """Pixel width of the widest caption that will be drawn, or 0 if there are none."""
    widest = 0
    for boxes in by_frame.values():
        for item in boxes:
            texts = [item.label]
            if item.decided and item.overlay is not None:
                texts.append(item.overlay.rule_id)
                if item.overlay.caveat:
                    texts.append(item.overlay.caveat)
            for text in texts:
                (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                widest = max(widest, int(w))
    return widest


def _even(value: float) -> int:
    """Round down to an even number, never below 2."""
    return max(2, int(value) - (int(value) % 2))


def _canvas(
    cv2: Any, frame: npt.NDArray[np.uint8], *, width: int, height: int
) -> npt.NDArray[np.uint8]:
    """The frame at output size, ready to draw on.

    Resized before annotation rather than after. Drawing first and shrinking afterwards
    would scale the captions down with the picture, which is precisely backwards: the
    reason to ask for a smaller file is to move it around, and text nobody can read
    makes the smaller file worthless.
    """
    if (frame.shape[1], frame.shape[0]) == (width, height):
        return np.ascontiguousarray(frame)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized).astype(np.uint8, copy=False)


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


def _draw_boxes(
    cv2: Any, canvas: npt.NDArray[np.uint8], boxes: Sequence[_Box], *, scale: float = 1.0
) -> int:
    """Draw one frame's boxes. Returns how many were drawn.

    `scale` matches the canvas: box coordinates are in source pixels, and the canvas may
    have been resized before this ran.
    """
    height = canvas.shape[0]
    captions = _Captions(height)
    # Top-down, so the stacking order on screen matches the order of the boxes down the
    # frame rather than the order tracks happen to have been created in.
    for item in sorted(boxes, key=lambda b: float(b.xyxy[1])):
        x1, y1, x2, y2 = (round(float(v) * scale) for v in item.xyxy)
        colour = item.colour
        left = max(0, x1 - 2)
        # Outside the box, never over it: a 1px border on a 10px target would cover a
        # fifth of the pixels the box is about, and small targets are the whole subject.
        cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), colour, 1)
        # White text, coloured box - and this survived the move to H.264, which the
        # first version of this comment predicted it would not. See CAPTION_COLOUR in
        # the module docstring: the damage to coloured text is chroma subsampling, not
        # the codec, and the luminance deficit is a property of the colour itself.
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
    """Draw text with a dark outline, so it stays readable against bright sky.

    Shifted left rather than allowed to run off the right edge. Captions keep their pixel
    size when the video is scaled down, which is deliberate - but it means a caption can
    be wider than a narrow frame, and text that runs off the edge is text the viewer
    silently does not get. The `NOT CLASSIFIED` line is exactly the one that would be
    lost, being the longest.
    """
    width = canvas.shape[1]
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x = max(0, min(origin[0], width - text_width - 2))
    placed = (x, origin[1])
    cv2.putText(canvas, text, placed, cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_COLOUR, 3, cv2.LINE_AA)
    cv2.putText(canvas, text, placed, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


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
