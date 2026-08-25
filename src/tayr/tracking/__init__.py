"""Tracking: Kalman filtering, detection-to-track association, and motion features.

SCOPE BOUNDARY. The Kalman filter's `predict` step advances the state by exactly one
frame, for the sole purpose of associating the next frame's detections with existing
tracks. There is no multi-step extrapolation API in this package and none may be added:
projecting a target's future position for the purpose of meeting it there is outside
this project's scope. See CLAUDE.md section 2.

Motion features answer "does this move like a bird or like a quadcopter", which is the
research question. That is the only purpose trajectory data serves here.
"""

from tayr.tracking.features import TrackFeatures, extract_features
from tayr.tracking.kalman import KalmanBoxFilter
from tayr.tracking.tracker import ByteTracker, Track, TrackState

__all__ = [
    "ByteTracker",
    "KalmanBoxFilter",
    "Track",
    "TrackFeatures",
    "TrackState",
    "extract_features",
]
