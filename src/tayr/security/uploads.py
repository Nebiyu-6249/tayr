"""Upload validation and media resource limits.

This is the highest-risk surface in Tayr. An uploaded video is attacker-controlled input
fed to `libav*`, which has a long history of memory-safety CVEs. Two distinct attacks
have to be defended separately:

**Type confusion.** A file's extension and its client-supplied MIME type are both just
strings the attacker chose. Trusting either lets `evil.mp4` actually be something else.
The container is therefore identified by reading its own magic bytes.

**Resource exhaustion.** A 30-second 16K video is a tiny file that expands to more RAM
than any host has. Decoded pixel count is the quantity that matters, and it is the
product of width, height, framerate and duration - so each is capped, and the product is
capped too. Every one of these is checked by **probing the container header before any
full decode**, which is the whole point: by the time a decoder has failed, it is too late.

Defence in depth: none of this replaces the worker sandbox. Validation reduces how often
a decoder sees hostile input; the sandbox is what contains it when validation is
bypassed. See docker-compose.yml.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

from tayr.errors import TayrError


class UploadRejectedError(TayrError):
    """An upload failed validation. The message is safe to show a user."""


@dataclass(frozen=True, slots=True)
class VideoLimits:
    """Caps enforced before decode.

    Defaults are deliberately conservative for an anonymous visitor. Owner-mode CLI work
    does not pass through this path at all.
    """

    max_bytes: int = 512 * 1024 * 1024
    max_duration_seconds: float = 300.0
    max_width: int = 4096
    max_height: int = 4096
    max_fps: float = 120.0
    max_total_pixels: float = 4e10
    """Width x height x fps x duration. The product is what exhausts memory: a 16K
    frame is fine on its own, and 300 seconds is fine on its own."""


# Container signatures, checked against the file's own bytes.
# ISO-BMFF (mp4/mov) carries 'ftyp' at offset 4 rather than a fixed magic at offset 0.
_ISO_BMFF_BRANDS = (b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"qt  ", b"M4V ", b"mmp4")
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x1a\x45\xdf\xa3", "matroska"),  # also webm
    (b"RIFF", "avi"),  # plus 'AVI ' at offset 8
)

ALLOWED_CONTAINERS = frozenset({"mp4", "matroska", "webm", "avi"})

_HEADER_BYTES = 32


def detect_container(header: bytes) -> str | None:
    """Identify a container from its leading bytes. None if unrecognised.

    Extension and client MIME type are never consulted: both are attacker-controlled.
    """
    if len(header) < 12:
        return None

    if header[4:8] == b"ftyp" and header[8:12] in _ISO_BMFF_BRANDS:
        return "mp4"

    for magic, name in _SIGNATURES:
        if header.startswith(magic):
            if name == "avi" and header[8:12] != b"AVI ":
                continue
            return name
    return None


def generate_storage_name(container: str) -> str:
    """A random storage filename.

    The user's filename is never used, in any part of a path. It is the classic path
    traversal vector ("../../etc/passwd", a NUL byte, a Windows device name), and it can
    also leak personal information into logs and URLs. The extension is derived from the
    detected container, not from what was uploaded.
    """
    if container not in ALLOWED_CONTAINERS:
        raise UploadRejectedError(f"unsupported container: {container}")
    extension = {"matroska": "mkv", "webm": "webm", "mp4": "mp4", "avi": "avi"}[container]
    return f"{secrets.token_urlsafe(24)}.{extension}"


def validate_upload(header: bytes, size_bytes: int, *, limits: VideoLimits | None = None) -> str:
    """Validate an upload's type and size. Returns the detected container name.

    `size_bytes` must be the number of bytes actually received. Enforce the cap while
    streaming as well: a Content-Length header is attacker-controlled, and checking only
    after the body has been buffered means the exhaustion already happened.
    """
    limits = limits or VideoLimits()

    if size_bytes <= 0:
        raise UploadRejectedError("empty upload")
    if size_bytes > limits.max_bytes:
        raise UploadRejectedError(
            f"file is {size_bytes / 1e6:.1f} MB; the limit is {limits.max_bytes / 1e6:.0f} MB"
        )

    container = detect_container(header[:_HEADER_BYTES])
    if container is None:
        raise UploadRejectedError(
            "unrecognised video container. Supported: MP4, MKV, WebM, AVI. "
            "The file's own contents are checked, not its name."
        )
    if container not in ALLOWED_CONTAINERS:
        raise UploadRejectedError(f"unsupported container: {container}")
    return container


@dataclass(frozen=True, slots=True)
class MediaProperties:
    """What a container probe reports. Populated by the worker via PyAV."""

    width: int
    height: int
    fps: float
    duration_seconds: float

    @property
    def total_pixels(self) -> float:
        return float(self.width) * self.height * self.fps * self.duration_seconds


def enforce_media_limits(props: MediaProperties, *, limits: VideoLimits | None = None) -> None:
    """Reject media exceeding any cap. Call after probing the header, before decoding.

    Raises on the first violation with a message naming the offending property, so a
    legitimate user can tell what to change.
    """
    limits = limits or VideoLimits()

    if props.width <= 0 or props.height <= 0:
        raise UploadRejectedError(f"invalid dimensions {props.width}x{props.height}")
    if props.fps <= 0:
        raise UploadRejectedError(f"invalid frame rate {props.fps}")
    if props.duration_seconds <= 0:
        raise UploadRejectedError(f"invalid duration {props.duration_seconds}s")

    if props.width > limits.max_width or props.height > limits.max_height:
        raise UploadRejectedError(
            f"resolution {props.width}x{props.height} exceeds the limit of "
            f"{limits.max_width}x{limits.max_height}"
        )
    if props.fps > limits.max_fps:
        raise UploadRejectedError(
            f"frame rate {props.fps:.1f} exceeds the limit of {limits.max_fps:.0f}"
        )
    if props.duration_seconds > limits.max_duration_seconds:
        raise UploadRejectedError(
            f"duration {props.duration_seconds:.1f}s exceeds the limit of "
            f"{limits.max_duration_seconds:.0f}s"
        )
    if props.total_pixels > limits.max_total_pixels:
        raise UploadRejectedError(
            f"total decoded pixel count {props.total_pixels:.3g} exceeds the limit of "
            f"{limits.max_total_pixels:.3g}. Each individual property is within its cap, "
            "but their product is not - this is the shape a decompression bomb takes."
        )


def is_safe_storage_path(path: Path, *, root: Path) -> bool:
    """True if `path` resolves inside `root`.

    Belt and braces alongside generated filenames: any future code path that builds a
    storage path from something a user influenced gets checked here too.
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved.is_relative_to(root_resolved)
