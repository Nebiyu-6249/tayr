"""DUT Anti-UAV converter - NOT IMPLEMENTED, and deliberately so.

The DUT Anti-UAV repository does not state its annotation format
`[VERIFIED: https://github.com/wangdongdut/DUT-Anti-UAV, accessed 2026-08-25 - the
README gives download links and a citation requirement but no format specification]`.

Writing a parser against a guessed format is the exact failure this project's converter
procedure exists to prevent: a wrong-but-plausible parser produces boxes that look fine
and trains a model that is quietly wrong.

TO IMPLEMENT THIS, first establish from the downloaded files:
  - field order and delimiter
  - coordinate convention: xyxy / xywh (top-left) / cxcywh (centre)
  - 0-indexed or 1-indexed
  - absolute pixels or normalised
  - how (or whether) an absent target is represented
  - the class id to name mapping
  - whether the detection and tracking subsets share a format (they may not - the
    detection subset is still images and the tracking subset is video)

Then read three real annotation files and confirm the spec matches what is on disk
before writing the parser, per `.claude/skills/dataset-converter`.

LICENCE: Apache-2.0 `[VERIFIED: repository sidebar]`. Citation required: Jie Zhao,
Jingshu Zhang, Dongdong Li, Dong Wang, "Vision-based Anti-UAV Detection and Tracking",
IEEE T-ITS 2022.

Documented split sizes are `[UNKNOWN]`. Secondary sources quote 10,000 detection images
split 5,200/2,600/2,200 and 20 tracking videos; the repository states none of this, so
those numbers must be confirmed by counting after download and must not be cited from
the secondary source.
"""

from __future__ import annotations

from tayr.datasets.schema import VideoAnnotation


def parse_dut_anti_uav(text: str, *, source_video: str) -> VideoAnnotation:
    """Not implemented. See the module docstring."""
    raise NotImplementedError(
        "The DUT Anti-UAV annotation format has not been established from a primary "
        "source - the repository README does not specify it. Download the dataset, "
        "inspect three annotation files, record the format in this module's docstring, "
        "and then implement the parser. Guessing the format here would produce "
        "plausible-looking boxes that are silently wrong. "
        f"(requested source_video={source_video!r}, {len(text)} chars of input)"
    )
