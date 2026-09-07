"""Render annotated frames to disk so boxes can be checked with an eye.

A converter with a sign error, a transposed axis, or a wrong index base still emits
boxes that pass every shape assertion and every round-trip test - the numbers are
self-consistent, they just sit in the wrong place. The only cheap detector for that is
looking at one.

**Two files per sample: the frame, and a zoom.** A 1920x1080 frame containing a 6px
target is unreadable at any size a person actually views it, so the full frame is written
for context and a crop around the box is written next to it, upscaled with
nearest-neighbour interpolation. Nearest-neighbour is not an aesthetic choice: bilinear
would blur the pixel boundary that a box-placement check is entirely about.

**The sample is chosen, not random.** Twelve random frames out of five thousand are
twelve medium boxes in open sky, which is the case least likely to be wrong. This picks
the frames that would expose a bug: the smallest box in the split (a one-pixel origin
error is 12% of an 8px box and visible), the largest, a frame carrying several objects,
and a frame carrying none - which should render with no box at all, and does not if the
converter is inventing them.

Each rendered frame is captioned with the box's pixels-on-target and its size bucket, so
a preview also answers "does this dataset actually contain the small targets the
research question is about" without a separate census run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tayr.datasets.schema import DatasetAnnotation, VideoAnnotation
from tayr.errors import ConfigError
from tayr.geometry import pixels_on_target, size_bucket

#: BGR, because OpenCV draws in BGR. Chosen to stay visible against sky.
BOX_COLOUR = (60, 240, 60)
TEXT_COLOUR = (255, 255, 255)
CAPTION_BACKDROP = (20, 20, 20)

#: How far outside a small box to draw, so a 5px box is visible at all.
BOX_PADDING = 6

#: How much context to show around a box in the zoom crop, as a multiple of its side.
CROP_CONTEXT = 4.0

#: The zoom crop is upscaled to roughly this many pixels on its long edge.
CROP_TARGET_PX = 480


@dataclass(frozen=True, slots=True)
class PreviewSample:
    """One frame chosen for rendering, and why it was chosen."""

    video: VideoAnnotation
    reason: str
    output_path: Path | None = None

    @property
    def file_name(self) -> str:
        return self.video.frames[0].file_name or f"{self.video.source_video}.jpg"


def _median_pot(video: VideoAnnotation) -> float:
    boxes = video.boxes_array()
    return float(np.median(pixels_on_target(boxes))) if len(boxes) else float("inf")


def choose_samples(dataset: DatasetAnnotation, *, limit: int = 12) -> list[PreviewSample]:
    """Pick frames that would expose a conversion bug, then fill up with a spread.

    Deterministic: same dataset in, same frames out, so two previews of the same split
    are comparable and a difference means the conversion changed.
    """
    if limit < 1:
        raise ConfigError(f"limit must be at least 1; got {limit}")

    single_frame = [v for v in dataset.videos if len(v.frames) == 1]
    if not single_frame:
        raise ConfigError(
            "preview renders one image per annotation and this dataset's videos hold "
            "multiple frames. Video datasets need frame extraction first."
        )

    with_boxes = [v for v in single_frame if v.n_boxes]
    empty = [v for v in single_frame if not v.n_boxes]
    multi = [v for v in single_frame if v.n_boxes > 1]

    chosen: dict[str, PreviewSample] = {}

    def take(video: VideoAnnotation | None, reason: str) -> None:
        if video is not None and video.source_video not in chosen:
            chosen[video.source_video] = PreviewSample(video=video, reason=reason)

    if with_boxes:
        by_size = sorted(with_boxes, key=_median_pot)
        take(by_size[0], "smallest box in the split")
        take(by_size[len(by_size) // 2], "median box size")
        take(by_size[-1], "largest box in the split")
    take(multi[0] if multi else None, "several objects in one frame")
    take(empty[0] if empty else None, "no objects: should render with no box drawn")

    # Fill the rest with an even sweep across the split rather than the first N, so a
    # preview covers the whole range of whatever varies along it.
    remaining = limit - len(chosen)
    if remaining > 0 and single_frame:
        step = max(1, len(single_frame) // (remaining + 1))
        for video in single_frame[::step]:
            if len(chosen) >= limit:
                break
            take(video, "sampled across the split")

    return list(chosen.values())[:limit]


def annotate_frame(image: Any, video: VideoAnnotation) -> Any:
    """Draw a video's boxes onto a copy of `image`, captioned with their size."""
    import cv2

    canvas = image.copy()
    caption = f"{video.source_video}   {video.n_boxes} box(es)"

    for obj in video.frames[0].objects:
        arr = obj.as_array()
        pot = float(pixels_on_target(arr))
        bucket = str(size_bucket(arr).item())
        x1, y1, x2, y2 = (round(v) for v in (obj.x1, obj.y1, obj.x2, obj.y2))
        # Draw outside the box, never over it: a 1px border on a 5px target would hide
        # a quarter of the pixels the annotation is about.
        cv2.rectangle(
            canvas,
            (x1 - BOX_PADDING, y1 - BOX_PADDING),
            (x2 + BOX_PADDING, y2 + BOX_PADDING),
            BOX_COLOUR,
            1,
        )
        cv2.putText(
            canvas,
            f"{obj.label.value} {pot:.1f}px {bucket}",
            (max(0, x1 - BOX_PADDING), max(12, y1 - BOX_PADDING - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BOX_COLOUR,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), CAPTION_BACKDROP, -1)
    cv2.putText(
        canvas, caption, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOUR, 1, cv2.LINE_AA
    )
    return canvas


def crop_around_box(image: Any, obj: Any, *, context: float = CROP_CONTEXT) -> Any:
    """A nearest-neighbour zoom on one box, drawn from the unannotated frame.

    Cropped from the raw image and upscaled first, then the box is drawn on the enlarged
    crop *exactly on the annotation boundary* - no padding, no label. At this zoom the
    border obscures nothing, and putting it on the true edge is the whole point: it shows
    which source pixels the annotation actually claims, which is what settles a question
    like whether the coordinates were 0-based or 1-based.

    Nearest-neighbour, so source pixels stay square and a one-pixel offset reads as one
    pixel rather than as a gradient.
    """
    import cv2

    height, width = image.shape[:2]
    side = max(obj.x2 - obj.x1, obj.y2 - obj.y1)
    half = max(8.0, side * (1.0 + context) / 2.0)
    cx, cy = (obj.x1 + obj.x2) / 2.0, (obj.y1 + obj.y2) / 2.0

    x1 = max(0, round(cx - half))
    y1 = max(0, round(cy - half))
    x2 = min(width, round(cx + half))
    y2 = min(height, round(cy + half))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        raise ConfigError(
            f"box ({obj.x1}, {obj.y1})-({obj.x2}, {obj.y2}) lies outside a "
            f"{width}x{height} image. That is a converter bug, not a rendering one."
        )

    scale = max(1, round(CROP_TARGET_PX / max(patch.shape[:2])))
    zoomed = cv2.resize(
        patch,
        (patch.shape[1] * scale, patch.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.rectangle(
        zoomed,
        (round((obj.x1 - x1) * scale), round((obj.y1 - y1) * scale)),
        (round((obj.x2 - x1) * scale) - 1, round((obj.y2 - y1) * scale) - 1),
        BOX_COLOUR,
        1,
    )
    return zoomed


def render_previews(
    dataset: DatasetAnnotation,
    *,
    image_root: Path,
    output_dir: Path,
    limit: int = 12,
    crops: bool = True,
) -> list[PreviewSample]:
    """Write annotated copies of chosen frames into `output_dir`.

    `image_root` is the split directory the `file_name` values are relative to. With
    `crops`, a zoomed view of each frame's first box is written beside the full frame.
    """
    import cv2

    if not image_root.is_dir():
        raise ConfigError(f"not a directory: {image_root}")

    samples = choose_samples(dataset, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[PreviewSample] = []
    for index, sample in enumerate(samples):
        source = image_root / sample.file_name
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ConfigError(
                f"could not read {source}. The annotation names it, so either the image "
                "is missing or the split directory passed here is the wrong one."
            )
        # PNG, not JPEG. The whole point of this output is a one-pixel border around a
        # five-pixel target, and JPEG's chroma subsampling smears exactly that.
        destination = output_dir / f"{index:02d}-{sample.video.source_video}.png"
        annotated = annotate_frame(image, sample.video)
        if not cv2.imwrite(str(destination), annotated):
            raise ConfigError(f"could not write {destination}")

        objects = sample.video.frames[0].objects
        if crops and objects:
            crop_path = destination.with_name(f"{destination.stem}-zoom.png")
            # From `image`, not `annotated`: the zoom draws its own exact box, and
            # cropping the annotated frame would drag the padded box and its label in.
            if not cv2.imwrite(str(crop_path), crop_around_box(image, objects[0])):
                raise ConfigError(f"could not write {crop_path}")
        rendered.append(
            PreviewSample(video=sample.video, reason=sample.reason, output_path=destination)
        )
    return rendered
