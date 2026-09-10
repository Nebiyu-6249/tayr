"""Annotated video output.

Optional and off by default: rendering re-decodes the whole video and re-encodes it,
which roughly doubles the wall time of a run and produces a file the headless path has
no use for. `tayr watch run --render` asks for it explicitly.

Nothing in here may state more than the decision record does. See `annotate.py`.
"""

from tayr.render.annotate import (
    CODEC_FALLBACKS,
    CRF_CAPABLE,
    FORMING_COLOUR,
    VERDICT_COLOURS,
    RenderResult,
    TrackOverlay,
    render_annotated_video,
    resolve_codec,
)

__all__ = [
    "CODEC_FALLBACKS",
    "CRF_CAPABLE",
    "FORMING_COLOUR",
    "VERDICT_COLOURS",
    "RenderResult",
    "TrackOverlay",
    "render_annotated_video",
    "resolve_codec",
]
