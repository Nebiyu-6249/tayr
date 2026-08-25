"""Pipeline orchestration: probe, decode, detect, track, extract features.

Ordering is a security property, not a preference:

    probe + enforce limits   <-- before a single frame is decoded
    decode frame by frame    <-- streaming, never the whole video in memory
    detect / track / feature

Progress is real. `frames_processed / frames_total` counts frames actually decoded
against a total estimated from the probe. Where the total cannot be determined, progress
reports 0 and the UI shows an indeterminate state rather than an invented percentage.

Any run using a detector whose `is_real` is False is marked synthetic, and that flag
travels all the way to the API response.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tayr.config import TrackerConfig
from tayr.errors import DependencyUnavailableError
from tayr.security.uploads import MediaProperties, VideoLimits
from tayr.tracking.features import MIN_OBSERVATIONS, TrackFeatures, extract_features
from tayr.tracking.tracker import ByteTracker, Track
from tayr.worker.detector import Detector
from tayr.worker.probe import probe_video

ProgressCallback = Callable[[int, int | None], None]


@dataclass(slots=True)
class PipelineResult:
    """What one job produced."""

    media: MediaProperties
    frames_processed: int
    frames_total: int | None
    tracks: list[Track] = field(default_factory=list)
    features: dict[int, TrackFeatures] = field(default_factory=dict)
    synthetic: bool = False
    detector_name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)


def iter_frames(path: Path, *, max_frames: int | None = None) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield RGB frames one at a time.

    Streaming, never a whole decoded video in memory: a 300-second 4K sequence is
    hundreds of gigabytes decoded, and materialising it would defeat every limit the
    probe just enforced.
    """
    try:
        import av
    except ImportError as exc:
        raise DependencyUnavailableError(
            "PyAV is required to decode video but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'"
        ) from exc

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        # Decode only what is needed; do not let libav spawn unbounded worker threads
        # inside a pids-capped container.
        stream.thread_type = "NONE"
        for count, frame in enumerate(container.decode(stream), start=1):
            # format="rgb24" always yields uint8; to_ndarray's annotation is a union
            # across every pixel format, so the narrowing is asserted here.
            rgb: npt.NDArray[np.uint8] = frame.to_ndarray(format="rgb24").astype(
                np.uint8, copy=False
            )
            yield rgb
            if max_frames is not None and count >= max_frames:
                return
    finally:
        container.close()


def run_pipeline(
    video_path: Path,
    detector: Detector,
    *,
    tracker_config: TrackerConfig | None = None,
    limits: VideoLimits | None = None,
    progress: ProgressCallback | None = None,
    max_frames: int | None = None,
) -> PipelineResult:
    """Run the full pipeline over one video.

    The probe runs first and raises before any decode if the media exceeds a limit.
    """
    # Step 1. Bound the media BEFORE decoding. This is the call that R1 was missing.
    media = probe_video(video_path, limits=limits)

    frames_total: int | None = None
    if media.fps > 0 and media.duration_seconds > 0:
        frames_total = round(media.fps * media.duration_seconds)
        if max_frames is not None:
            frames_total = min(frames_total, max_frames)

    tracker = ByteTracker(tracker_config or TrackerConfig())
    frames_processed = 0

    # Step 2. Decode and process, one frame at a time.
    for frame_index, frame in enumerate(iter_frames(video_path, max_frames=max_frames)):
        detection = detector.detect(frame)
        tracker.update(detection.boxes_xyxy, detection.scores, frame_index=frame_index)
        frames_processed += 1
        if progress is not None:
            progress(frames_processed, frames_total)

    tracks = tracker.finalise()

    # Step 3. Motion features, for tracks long enough to support them.
    features: dict[int, TrackFeatures] = {}
    notes: list[str] = []
    too_short = 0
    for track in tracks:
        if track.n_observations < MIN_OBSERVATIONS:
            too_short += 1
            continue
        features[track.track_id] = extract_features(
            np.stack(track.observed_boxes),
            np.array(track.observed_frames, dtype=np.int64),
            fps=media.fps if media.fps > 0 else 30.0,
        )
    if too_short:
        notes.append(
            f"{too_short} track(s) had fewer than {MIN_OBSERVATIONS} observations and "
            "carry no motion features; a shorter track cannot support an acceleration "
            "variance or a dominant frequency."
        )

    synthetic = not detector.is_real
    if synthetic:
        notes.append(
            f"SYNTHETIC: detector {detector.name!r} is a stand-in, not a trained model. "
            "No number in this result describes real-world performance."
        )

    return PipelineResult(
        media=media,
        frames_processed=frames_processed,
        frames_total=frames_total,
        tracks=tracks,
        features=features,
        synthetic=synthetic,
        detector_name=detector.name,
        notes=notes,
    )
