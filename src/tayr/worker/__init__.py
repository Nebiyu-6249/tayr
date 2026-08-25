"""Worker: the sandboxed process that decodes video and runs the pipeline.

This is the only place in Tayr where attacker-supplied media is parsed. Everything here
runs inside the container described in docker-compose.yml: unprivileged, read-only root
filesystem, all capabilities dropped, no network egress, and hard resource caps.
"""

from tayr.worker.detector import Detection, Detector, StubDetector
from tayr.worker.pipeline import PipelineResult, run_pipeline
from tayr.worker.probe import probe_video

__all__ = [
    "Detection",
    "Detector",
    "PipelineResult",
    "StubDetector",
    "probe_video",
    "run_pipeline",
]
