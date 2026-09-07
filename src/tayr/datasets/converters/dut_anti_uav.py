"""DUT Anti-UAV: what is implemented, and what is deliberately not.

The dataset ships two subsets with different shapes, and they need different answers.

## Detection subset — implemented, as `voc`

Pascal VOC XML, one file per image, in `<split>/xml/` beside `<split>/img/`
`[VERIFIED: sample annotation from the downloaded dataset, and the layout reported by
the dataset holder, 2026-09-07]`. Parsed by `tayr.datasets.converters.voc`; use
`--format voc` on `tayr dataset census`, `convert`, `prepare` and `preview`.

There are no tracks in it. Each image is an independent still, so every image becomes
its own single-frame group and `take_census` reports `tracks 0` along with the reason.
Nothing infers a track from filename order: consecutively numbered stills are not a
sequence, and treating them as one would fabricate exactly the track data this project
does not have. See docs/RESEARCH.md 14.4.

## Tracking subset — NOT implemented, because there is nothing to parse

`Anti-UAV-Tracking-V0` is 20 videos and 24,804 frames, and ships one
`videoNN_gt_first.txt` per video holding a **single first-frame box**
`[REPORTED BY THE DATASET HOLDER, 2026-09-07; not verified in this environment]`.

That is 20 initialisation boxes and zero annotated tracks - roughly 0.08% of frames
carry a label. A converter for it would parse 20 numbers and hand back 20 one-frame
"tracks", which is not a converter failure but a data one, and writing the parser would
disguise it. The first-frame box is genuinely useful, just not as annotation: it is the
initialisation anchor for a tracker-derived pseudo-track, and the only checkable point
in one. docs/RESEARCH.md 14.4 covers what that does and does not let you claim.

Implement a reader for it when the pseudo-track pipeline needs one - it belongs with
that pipeline, where the first-frame box is an input, rather than here among the
annotation converters, where it would look like ground truth.

LICENCE: Apache-2.0 `[VERIFIED: repository sidebar, Phase 0]`. Citation required: Jie
Zhao, Jingshu Zhang, Dongdong Li, Dong Wang, "Vision-based Anti-UAV Detection and
Tracking", IEEE T-ITS 2022.

Split sizes are `[UNKNOWN]` from the repository, which states none. The holder reports
5,200 / 2,600 / 2,200 detection images and 20 tracking videos; confirm by counting after
download - `tayr dataset census` prints what is actually there - and cite the count, not
the claim.
"""

from __future__ import annotations

from tayr.datasets.schema import VideoAnnotation


def parse_dut_tracking_gt_first(text: str, *, source_video: str) -> VideoAnnotation:
    """Not implemented. See the module docstring.

    Named for what the file actually is - a first-frame box - rather than for the
    dataset, so no caller can mistake it for per-frame ground truth.
    """
    raise NotImplementedError(
        "DUT Anti-UAV's tracking subset ships only a first-frame box per video: 20 "
        "boxes across 24,804 frames, with no per-frame ground truth. There is no track "
        "annotation to convert, and returning 20 one-frame tracks would disguise a data "
        "gap as a conversion result. The detection subset IS supported - use "
        "--format voc. See docs/RESEARCH.md 14.4 for the pseudo-track route this box is "
        f"actually an input to. (requested source_video={source_video!r}, "
        f"{len(text)} chars of input)"
    )
