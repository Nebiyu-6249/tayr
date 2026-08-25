"""Anti-UAV annotation converter (IR_label.json family).

FORMAT, established from the benchmark's own evaluation toolkit source rather than
from prose:

  - File: one `IR_label.json` per sequence directory
    `[VERIFIED: https://raw.githubusercontent.com/HwangBo94/Anti-UAV410/main/datasets/antiuav410.py
     line 38, glob '*/IR_label.json']`
  - Key `gt_rect`: a list with **one entry per image file**, each a 4-element box
    `[VERIFIED: same file lines 66-67 -
      assert len(img_files) == len(label_res['gt_rect'])
      assert len(label_res['gt_rect'][0]) == 4]`
  - Coordinate convention: **`(left, top, width, height)`**
    `[VERIFIED: https://raw.githubusercontent.com/HwangBo94/Anti-UAV410/main/utils/metrics.py
     lines 11-12 - "each line represent a rectangle (left, top, width, height)"]`
  - Key `exist`: **not read by the Anti-UAV410 loader** `[VERIFIED: absent from
    antiuav410.py]`. Other releases in this family (Anti-UAV300/600) are reported to
    carry a per-frame `exist` flag of 1/0 `[UNKNOWN - not confirmed against a primary
    source]`. The parser therefore honours `exist` **when present** and otherwise
    falls back to treating an empty rect as an absent target. Which rule fired is
    reported, not assumed.
  - Track ids: this is a **single-target** tracking benchmark, so one sequence is one
    track. That identity is inferred, not annotated, and is recorded as
    `TrackIdSource.SINGLE_TARGET` so the census can separate it from real annotations.
  - Class: sequences contain a UAV only; every box is `ObjectClass.DRONE`.

MODALITY WARNING: Anti-UAV410 and Anti-UAV600 are **IR-only**; only Anti-UAV300
contains RGB video `[VERIFIED: https://github.com/ZhaoJ9014/Anti-UAV README - "410 and
600 versions only contain IR videos while 300 version contains both RGB videos and IR
videos"]`. An RGB pipeline can only use the 300 release.

LICENCE: MIT `[VERIFIED: https://github.com/ZhaoJ9014/Anti-UAV - "The project of
Anti-UAV is released under the MIT License"]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tayr.datasets.schema import (
    BoxAnnotation,
    FrameAnnotation,
    ObjectClass,
    TrackIdSource,
    VideoAnnotation,
)
from tayr.errors import ConfigError


@dataclass(frozen=True, slots=True)
class AntiUavParseReport:
    """Which absence rule actually fired, so the assumption is visible in the census."""

    used_exist_key: bool
    n_frames: int
    n_present: int
    n_absent: int


def parse_anti_uav_json(
    text: str, *, source_video: str, track_id: int = 0
) -> tuple[VideoAnnotation, AntiUavParseReport]:
    """Parse one IR_label.json into canonical form.

    Absent-target frames become frames with zero objects. They are never dropped:
    doing so would destroy the denominator for false-alarm rate.
    """
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source_video}: IR_label.json is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{source_video}: expected a JSON object, got {type(data).__name__}")
    if "gt_rect" not in data:
        raise ConfigError(
            f"{source_video}: no 'gt_rect' key. The Anti-UAV toolkit requires it "
            "(datasets/antiuav410.py asserts on it), so this file is not in the "
            "expected format."
        )

    rects = data["gt_rect"]
    if not isinstance(rects, list):
        raise ConfigError(f"{source_video}: 'gt_rect' must be a list, got {type(rects).__name__}")

    raw_exist = data.get("exist")
    exist: list[Any] | None = raw_exist if isinstance(raw_exist, list) else None
    used_exist = exist is not None
    if exist is not None and len(exist) != len(rects):
        raise ConfigError(
            f"{source_video}: 'exist' has {len(exist)} entries but 'gt_rect' has "
            f"{len(rects)}. They must correspond one-to-one."
        )

    frames: list[FrameAnnotation] = []
    n_present = 0

    for i, rect in enumerate(rects):
        present = bool(exist[i]) if exist is not None else _rect_is_present(rect)

        if not present:
            frames.append(FrameAnnotation(frame_index=i, objects=()))
            continue

        if not isinstance(rect, list) or len(rect) != 4:
            raise ConfigError(
                f"{source_video}: frame {i} is marked present but its rect is {rect!r}; "
                "expected 4 numbers (left, top, width, height)."
            )
        try:
            left, top, width, height = (float(v) for v in rect)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source_video}: frame {i} has non-numeric rect {rect!r}") from exc

        if width <= 0 or height <= 0:
            raise ConfigError(
                f"{source_video}: frame {i} is marked present but has non-positive size "
                f"w={width} h={height}. Not clamped - this is a dataset or parser bug."
            )

        frames.append(
            FrameAnnotation(
                frame_index=i,
                objects=(
                    BoxAnnotation(
                        x1=left,
                        y1=top,
                        x2=left + width,
                        y2=top + height,
                        label=ObjectClass.DRONE,
                        track_id=track_id,
                    ),
                ),
            )
        )
        n_present += 1

    video = VideoAnnotation(
        source_video=source_video,
        frames=tuple(frames),
        track_id_source=TrackIdSource.SINGLE_TARGET,
        frame_index_base=0,
    )
    report = AntiUavParseReport(
        used_exist_key=used_exist,
        n_frames=len(frames),
        n_present=n_present,
        n_absent=len(frames) - n_present,
    )
    return video, report


def _rect_is_present(rect: object) -> bool:
    """Fallback absence rule used only when the file has no 'exist' key.

    An empty list means absent. An all-zero rect is also treated as absent, which is a
    common encoding in this dataset family - and a zero-size box would be rejected by
    `BoxAnnotation` anyway, so the alternative is a crash rather than a silent error.
    """
    if not isinstance(rect, list) or len(rect) != 4:
        return False
    try:
        values = [float(v) for v in rect]
    except (TypeError, ValueError):
        return False
    return any(v != 0.0 for v in values)


def write_anti_uav_json(video: VideoAnnotation, *, include_exist: bool = True) -> str:
    """Serialise canonical form back to IR_label.json. Inverse of the parser."""
    rects: list[list[float]] = []
    exist: list[int] = []

    for frame in video.frames:
        if frame.is_empty:
            rects.append([])
            exist.append(0)
            continue
        if len(frame.objects) != 1:
            raise ConfigError(
                f"{video.source_video}: frame {frame.frame_index} has "
                f"{len(frame.objects)} objects. Anti-UAV is single-target and its "
                "format cannot represent more than one."
            )
        obj = frame.objects[0]
        rects.append([obj.x1, obj.y1, obj.x2 - obj.x1, obj.y2 - obj.y1])
        exist.append(1)

    payload: dict[str, Any] = {"gt_rect": rects}
    if include_exist:
        payload["exist"] = exist
    return json.dumps(payload)
