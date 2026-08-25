"""Motion features extracted from a track.

These are the input to the motion arm of Tayr's hypothesis: that below roughly 20 pixels
on target, appearance saturates while motion signature stays discriminative.

Each feature is a scalar computed over a track's **observed** history. Frames where no
detection was associated are not interpolated - filling them with filter predictions
would manufacture smooth motion that was never measured, and smoothness is one of the
things being measured.

The features, and the physical intuition each is meant to capture:

  `mean_speed`, `speed_variance`   A quadcopter in transit holds speed; a bird's flapping
                                   flight modulates it.
  `acceleration_variance`          Birds accelerate erratically; multirotors change
                                   velocity smoothly under control.
  `vertical_oscillation_hz`        Dominant frequency of the vertical position signal.
                                   Flapping flight has one; powered flight does not.
  `vertical_oscillation_power`     How much of the signal that frequency accounts for.
                                   A dominant frequency with negligible power is noise.
  `heading_entropy`                Spread of direction changes. Purposeful transit is
                                   low-entropy; foraging or soaring is high.
  `hover_fraction`                 Proportion of the track spent below a speed threshold.
                                   Multirotors hover; fixed-wing aircraft and most birds
                                   cannot.
  `trajectory_smoothness`          Path length divided by straight-line displacement.
                                   1.0 is a perfectly straight path.
  `scale_change_rate`              Median per-frame fractional change in sqrt(area), a
                                   proxy for range change.

NOT A CLASSIFIER OF INTENT, AND NOT A PREDICTOR. These describe motion already observed,
to answer "what kind of thing is this". Nothing here extrapolates a future position. See
the package docstring and CLAUDE.md section 2.

HONESTY NOTE: none of these has been validated against real bird or drone tracks - as of
Phase 4 no such data is in hand (docs/RESEARCH.md 14.4). The tests below assert only that
each feature responds in the expected direction on synthetic trajectories constructed to
have the property being measured. That is a correctness check on the arithmetic, not
evidence that these features separate real classes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from tayr.errors import GeometryError
from tayr.geometry import FloatArray, pixels_on_target, xyxy_to_cxcywh

# Below this many observations, the derived quantities (acceleration, dominant frequency)
# have too few samples to mean anything.
MIN_OBSERVATIONS = 8

# Speed below this many pixels per frame counts as hovering. A real value must come from
# measured data; this default is a placeholder and is reported as one.
HOVER_SPEED_PX_PER_FRAME = 0.5


@dataclass(frozen=True, slots=True)
class TrackFeatures:
    """Scalar motion descriptors for one track."""

    n_observations: int
    duration_frames: int
    mean_speed: float
    speed_variance: float
    acceleration_variance: float
    vertical_oscillation_hz: float
    vertical_oscillation_power: float
    heading_entropy: float
    hover_fraction: float
    trajectory_smoothness: float
    scale_change_rate: float
    median_pixels_on_target: float

    def as_vector(self) -> FloatArray:
        """Feature vector for the classifier, in a fixed documented order."""
        return np.array(
            [
                self.mean_speed,
                self.speed_variance,
                self.acceleration_variance,
                self.vertical_oscillation_hz,
                self.vertical_oscillation_power,
                self.heading_entropy,
                self.hover_fraction,
                self.trajectory_smoothness,
                self.scale_change_rate,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "mean_speed",
            "speed_variance",
            "acceleration_variance",
            "vertical_oscillation_hz",
            "vertical_oscillation_power",
            "heading_entropy",
            "hover_fraction",
            "trajectory_smoothness",
            "scale_change_rate",
        ]

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def extract_features(
    boxes_xyxy: FloatArray,
    frame_indices: npt.NDArray[np.int64] | list[int],
    *,
    fps: float = 30.0,
    hover_speed: float = HOVER_SPEED_PX_PER_FRAME,
) -> TrackFeatures:
    """Compute motion features for one track's observed history.

    `frame_indices` must be the frames where a detection was actually associated. Gaps
    are respected: speeds are divided by the real frame delta, so a track that was lost
    for ten frames does not register a spurious jump.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)

    if len(boxes) != len(frames):
        raise GeometryError(f"{len(boxes)} box(es) but {len(frames)} frame index(es)")
    if len(boxes) < MIN_OBSERVATIONS:
        raise GeometryError(
            f"need at least {MIN_OBSERVATIONS} observations to derive motion features; "
            f"got {len(boxes)}. A shorter track cannot support an acceleration variance "
            "or a dominant frequency, and returning one anyway would be a fabricated number."
        )
    if np.any(np.diff(frames) <= 0):
        raise GeometryError("frame_indices must be strictly increasing")
    if fps <= 0:
        raise GeometryError(f"fps must be positive; got {fps}")

    centres = xyxy_to_cxcywh(boxes)[:, :2]
    dt = np.diff(frames).astype(np.float64)
    deltas = np.diff(centres, axis=0)

    velocities = deltas / dt[:, None]
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])

    # Acceleration over possibly-uneven spacing.
    if len(velocities) >= 2:
        dt_mid = (dt[:-1] + dt[1:]) / 2.0
        accelerations = np.diff(velocities, axis=0) / dt_mid[:, None]
        accel_var = float(np.var(np.hypot(accelerations[:, 0], accelerations[:, 1])))
    else:
        accel_var = 0.0

    osc_hz, osc_power = _dominant_frequency(centres[:, 1], frames, fps=fps)

    straight_line = float(np.linalg.norm(centres[-1] - centres[0]))
    path_length = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))
    # A track that returns to its origin has zero displacement; report the path length
    # ratio against a floor rather than dividing by zero.
    smoothness = path_length / max(straight_line, 1e-6)

    pot = pixels_on_target(boxes)
    scale_ratios = pot[1:] / np.maximum(pot[:-1], 1e-6)
    scale_rate = float(np.median(np.abs(scale_ratios - 1.0) / dt))

    return TrackFeatures(
        n_observations=len(boxes),
        duration_frames=int(frames[-1] - frames[0] + 1),
        mean_speed=float(np.mean(speeds)),
        speed_variance=float(np.var(speeds)),
        acceleration_variance=accel_var,
        vertical_oscillation_hz=osc_hz,
        vertical_oscillation_power=osc_power,
        heading_entropy=_heading_entropy(velocities),
        hover_fraction=float(np.mean(speeds < hover_speed)),
        trajectory_smoothness=float(smoothness),
        scale_change_rate=scale_rate,
        median_pixels_on_target=float(np.median(pot)),
    )


def _dominant_frequency(
    signal: FloatArray, frames: npt.NDArray[np.int64], *, fps: float
) -> tuple[float, float]:
    """Dominant frequency of a position signal, in Hz, plus its share of total power.

    The linear trend is removed first: a target flying steadily downward has a huge
    low-frequency component that would otherwise swamp any real oscillation.

    Gaps are resampled onto a uniform grid, because an FFT over unevenly-spaced samples
    reports frequencies that are not there. The power share is returned alongside the
    frequency so a caller can tell a genuine oscillation from the loudest bin of noise -
    a dominant frequency with negligible power means nothing.
    """
    n_uniform = int(frames[-1] - frames[0]) + 1
    if n_uniform < MIN_OBSERVATIONS:
        return 0.0, 0.0

    grid = np.arange(frames[0], frames[-1] + 1, dtype=np.float64)
    resampled = np.interp(grid, frames.astype(np.float64), signal)

    detrended = resampled - np.poly1d(np.polyfit(grid, resampled, 1))(grid)
    if np.allclose(detrended, 0.0):
        return 0.0, 0.0

    windowed = detrended * np.hanning(len(detrended))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / fps)

    # Bin 0 is the DC term, which detrending has already handled.
    if len(spectrum) < 2:
        return 0.0, 0.0
    peak = int(np.argmax(spectrum[1:])) + 1
    total = float(np.sum(spectrum[1:]))
    share = float(spectrum[peak] / total) if total > 0 else 0.0
    return float(freqs[peak]), share


def _heading_entropy(velocities: FloatArray) -> float:
    """Shannon entropy of heading direction over 8 compass bins, normalised to [0, 1].

    Stationary steps carry no heading and are excluded: including them would put a
    hovering target's noise into the direction histogram and read as high entropy.
    """
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    moving = speeds > 1e-9
    if moving.sum() < 2:
        return 0.0

    headings = np.arctan2(velocities[moving, 1], velocities[moving, 0])
    n_bins = 8
    counts, _ = np.histogram(headings, bins=n_bins, range=(-math.pi, math.pi))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return entropy / math.log2(n_bins)
