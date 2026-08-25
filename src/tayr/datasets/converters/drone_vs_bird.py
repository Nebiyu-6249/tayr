"""Drone-vs-Bird (WOSDETC challenge) annotation converter.

FORMAT, quoted verbatim from the challenge repository README
`[VERIFIED: https://github.com/wosdetc/challenge, accessed 2026-08-25]`:

    framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class ...

One whitespace-delimited text file per video.

  - Coordinates: **top-left `xywh`**, absolute pixels `[VERIFIED: field names
    x_left / y_top / w / h in the spec above]`
  - Class: present as a trailing field per object `[VERIFIED: obj1_class in the spec]`
  - Frame index base: **`[UNKNOWN]`** - the README does not say whether `framenum`
    starts at 0 or 1. `parse_drone_vs_bird` therefore records the minimum observed
    index as `frame_index_base` rather than assuming, and round-trips it.
  - Track ids: **not present in the format.** Objects carry no identity across frames,
    so `track_id` is None and `TrackIdSource.NONE` is recorded. This matters for the
    census: Drone-vs-Bird contributes detection boxes but no annotated tracks, and the
    motion arm of the hypothesis needs tracks.

LICENCE: distributed under a data usage agreement obtained by emailing
`wosdetc@googlegroups.com`; use is restricted to research purposes and **no
redistribution rights are granted** `[VERIFIED: same README]`. Never commit the raw
annotation files, converted annotation files, or any derived frames.
"""

from __future__ import annotations

from tayr.datasets.schema import (
    BoxAnnotation,
    FrameAnnotation,
    ObjectClass,
    TrackIdSource,
    VideoAnnotation,
)
from tayr.errors import ConfigError

# The challenge's class tokens are not enumerated in the README, so the mapping is
# built defensively: anything unrecognised becomes UNKNOWN and is counted, never
# silently coerced to DRONE. `[UNKNOWN: the authoritative class vocabulary]`
_CLASS_MAP: dict[str, ObjectClass] = {
    "drone": ObjectClass.DRONE,
    "uav": ObjectClass.DRONE,
    "bird": ObjectClass.BIRD,
    "airplane": ObjectClass.AIRCRAFT,
    "aircraft": ObjectClass.AIRCRAFT,
    "helicopter": ObjectClass.AIRCRAFT,
}

_FIELDS_PER_OBJECT = 5  # x_left, y_top, w, h, class


def _map_class(token: str) -> ObjectClass:
    return _CLASS_MAP.get(token.strip().lower(), ObjectClass.UNKNOWN)


def parse_drone_vs_bird(text: str, *, source_video: str) -> VideoAnnotation:
    """Parse one Drone-vs-Bird annotation file into canonical form.

    Raises ConfigError with the line number on any structural problem. A malformed
    annotation line is a real error: silently skipping it would quietly shrink the
    dataset and change every metric computed from it.
    """
    frames: list[FrameAnnotation] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ConfigError(
                f"{source_video}:{lineno}: expected at least 'framenum num_objs', got {line!r}"
            )

        try:
            frame_index = int(parts[0])
            n_objs = int(parts[1])
        except ValueError as exc:
            raise ConfigError(
                f"{source_video}:{lineno}: framenum and num_objs must be integers, got {line!r}"
            ) from exc

        if n_objs < 0:
            raise ConfigError(f"{source_video}:{lineno}: negative object count {n_objs}")

        expected = 2 + n_objs * _FIELDS_PER_OBJECT
        if len(parts) != expected:
            raise ConfigError(
                f"{source_video}:{lineno}: declares {n_objs} object(s) so expected "
                f"{expected} fields, found {len(parts)}. Line: {line!r}"
            )

        objects: list[BoxAnnotation] = []
        for i in range(n_objs):
            base = 2 + i * _FIELDS_PER_OBJECT
            try:
                x, y, w, h = (float(parts[base + j]) for j in range(4))
            except ValueError as exc:
                raise ConfigError(
                    f"{source_video}:{lineno}: object {i} has non-numeric box fields"
                ) from exc

            if w <= 0 or h <= 0:
                raise ConfigError(
                    f"{source_video}:{lineno}: object {i} has non-positive size w={w} h={h}. "
                    "This is a dataset or parser bug; it is not clamped."
                )

            objects.append(
                BoxAnnotation(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    label=_map_class(parts[base + 4]),
                    track_id=None,  # format carries no track identity
                )
            )

        frames.append(FrameAnnotation(frame_index=frame_index, objects=tuple(objects)))

    frames.sort(key=lambda f: f.frame_index)
    base = min((f.frame_index for f in frames), default=0)

    return VideoAnnotation(
        source_video=source_video,
        frames=tuple(frames),
        track_id_source=TrackIdSource.NONE,
        frame_index_base=base,
    )


def write_drone_vs_bird(video: VideoAnnotation) -> str:
    """Serialise canonical form back to the native format. Inverse of the parser.

    Exists so the round-trip test can prove the parser did not lose or transpose a
    field. Class tokens are written back in the dataset's own vocabulary.
    """
    reverse = {
        ObjectClass.DRONE: "drone",
        ObjectClass.BIRD: "bird",
        ObjectClass.AIRCRAFT: "aircraft",
        ObjectClass.UNKNOWN: "unknown",
    }

    lines: list[str] = []
    for frame in video.frames:
        parts: list[str] = [str(frame.frame_index), str(len(frame.objects))]
        for obj in frame.objects:
            parts += [
                _fmt(obj.x1),
                _fmt(obj.y1),
                _fmt(obj.x2 - obj.x1),
                _fmt(obj.y2 - obj.y1),
                reverse[obj.label],
            ]
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def _fmt(value: float) -> str:
    """Integral values print without a trailing '.0' so the round-trip is textual."""
    return str(int(value)) if float(value).is_integer() else repr(value)
