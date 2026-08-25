"""Container probing.

**This module closes the gap recorded as R1 in docs/THREAT_MODEL.md.**
`enforce_media_limits` existed and was tested from Phase 7, but nothing called it. It is
called here, on the container header, *before* any frame is decoded.

Order matters and is not negotiable:

    1. open the container and read its header only
    2. enforce every limit
    3. only then decode frames

Reversing 2 and 3 makes the limits decorative: by the time a decoder has consumed a
decompression bomb, the memory is already gone. PyAV is used rather than shelling out to
an `ffmpeg` binary precisely because it exposes stream metadata without decoding, and
because there is no subprocess for an attacker to influence.

A file whose header parses but whose streams are absent, or whose duration is unknown,
is rejected rather than guessed at. An unknown duration cannot be bounded, so it cannot
be allowed.
"""

from __future__ import annotations

from pathlib import Path

from tayr.errors import DependencyUnavailableError
from tayr.security.uploads import (
    MediaProperties,
    UploadRejectedError,
    VideoLimits,
    enforce_media_limits,
)

# Guard against a container that declares an absurd frame rate in its header.
_MAX_PLAUSIBLE_FPS = 1000.0


def probe_video(path: Path, *, limits: VideoLimits | None = None) -> MediaProperties:
    """Read a container's header, enforce the limits, and return its properties.

    Raises `UploadRejectedError` if the file is unreadable, has no video stream, has an
    indeterminate duration, or exceeds any limit. Raises `DependencyUnavailableError` if
    PyAV is not installed - never a silent fallback, because a fallback here would mean
    decoding unbounded media.
    """
    try:
        import av
    except ImportError as exc:
        raise DependencyUnavailableError(
            "PyAV is required to probe video but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'. Tayr will not decode media "
            "without the ability to bound it first."
        ) from exc

    if not path.is_file():
        raise UploadRejectedError(f"not a file: {path.name}")

    try:
        container = av.open(str(path))
    except Exception as exc:
        # Any libav failure at open time. The message is deliberately generic: a libav
        # error string can echo file contents back to the uploader.
        raise UploadRejectedError("could not read this file as video") from exc

    try:
        streams = container.streams.video
        if not streams:
            raise UploadRejectedError("file contains no video stream")

        stream = streams[0]

        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)

        fps = 0.0
        if stream.average_rate:
            fps = float(stream.average_rate)
        elif stream.guessed_rate:
            fps = float(stream.guessed_rate)
        if fps > _MAX_PLAUSIBLE_FPS:
            raise UploadRejectedError(
                f"declared frame rate {fps:.0f} is not plausible; the container header "
                "is malformed or hostile"
            )

        duration = _duration_seconds(stream, container)
        if duration is None:
            raise UploadRejectedError(
                "duration could not be determined from the container. An unbounded "
                "duration cannot be limited, so the file is refused."
            )

        properties = MediaProperties(width=width, height=height, fps=fps, duration_seconds=duration)
        # The limits actually run here. Nothing has been decoded at this point.
        enforce_media_limits(properties, limits=limits)
        return properties
    finally:
        container.close()


def _duration_seconds(stream: object, container: object) -> float | None:
    """Duration in seconds from the stream, falling back to the container.

    Returns None rather than a guess. Both values are optional in most containers, and
    a fabricated duration would defeat the duration cap.
    """
    stream_duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if stream_duration and time_base:
        return float(stream_duration * time_base)

    container_duration = getattr(container, "duration", None)
    if container_duration:
        # Container duration is in AV_TIME_BASE units (microseconds).
        return float(container_duration) / 1_000_000.0
    return None
