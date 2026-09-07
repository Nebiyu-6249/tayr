"""Export canonical annotations to COCO detection format.

COCO is what the detector consumes. This is a one-way export: the canonical form is
richer (it keeps track identity, empty frames, and video grouping), so round-tripping
through COCO would lose exactly the fields the research question depends on.

Two COCO conventions to keep straight, both of which are easy to get wrong:
  - COCO `bbox` is **`[x, y, width, height]`** top-left, not xyxy.
  - COCO `category_id` is conventionally 1-based; 0 is reserved for background.

`images` includes frames with no annotations. Dropping them would remove the negative
examples that false-alarm rate is computed against.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tayr.datasets.schema import DatasetAnnotation, ObjectClass
from tayr.errors import ConfigError
from tayr.geometry import FloatArray, xywh_to_xyxy

# 1-based, per COCO convention. Stable across exports so category ids do not shift
# between runs and silently invalidate a comparison.
CATEGORY_IDS: dict[ObjectClass, int] = {
    ObjectClass.DRONE: 1,
    ObjectClass.BIRD: 2,
    ObjectClass.AIRCRAFT: 3,
    ObjectClass.UNKNOWN: 4,
}


def to_coco(dataset: DatasetAnnotation) -> dict[str, Any]:
    """Build a COCO-format dict from canonical annotations.

    `images` carries a `source_video` field beyond the COCO spec, so that
    grouped-by-video splitting stays possible after export. Standard COCO tooling
    ignores unknown keys.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 0
    ann_id = 0

    for video in dataset.videos:
        for frame in video.frames:
            image_id += 1
            images.append(
                {
                    "id": image_id,
                    # A real filename wins. Only video datasets, which address frames
                    # by (video, index) rather than by name, fall through to the
                    # synthesised form.
                    "file_name": frame.file_name
                    or f"{video.source_video}/{frame.frame_index:06d}.jpg",
                    "width": video.width,
                    "height": video.height,
                    # Non-standard, deliberately: evaluation splits are grouped by video
                    # and that grouping must survive the export.
                    "source_video": video.source_video,
                    "frame_index": frame.frame_index,
                }
            )
            for obj in frame.objects:
                ann_id += 1
                w = obj.x2 - obj.x1
                h = obj.y2 - obj.y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_IDS[obj.label],
                        "bbox": [obj.x1, obj.y1, w, h],  # COCO is xywh, not xyxy
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": obj.track_id,
                    }
                )

    return {
        "info": {
            "description": f"{dataset.name} [{dataset.split}] exported by Tayr",
            "licence": dataset.licence,
            "redistributable": dataset.redistributable,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": cls.value}
            for cls, cid in sorted(CATEGORY_IDS.items(), key=lambda kv: kv[1])
        ],
    }


@dataclass(frozen=True, slots=True)
class CocoImage:
    """One image from a COCO annotation file, with its ground truth in xyxy."""

    image_id: int
    file_name: str
    width: int
    height: int
    boxes_xyxy: FloatArray
    category_ids: npt.NDArray[np.int64]
    source_video: str | None = None
    """Non-standard field written by `to_coco`. Present, evaluation splits can stay
    grouped by video; absent, they cannot, and the caller needs to know which."""


@dataclass(frozen=True, slots=True)
class CocoSplit:
    """A parsed COCO detection split."""

    path: Path
    images: list[CocoImage]
    categories: dict[int, str]

    @property
    def n_boxes(self) -> int:
        return sum(len(image.boxes_xyxy) for image in self.images)

    @property
    def has_video_grouping(self) -> bool:
        return all(image.source_video is not None for image in self.images)


def read_coco_split(path: Path) -> CocoSplit:
    """Read a COCO detection JSON back into boxes.

    The inverse of `to_coco` only for the detection view: track identity and video
    structure do not survive a COCO round trip, which is why `to_coco` is documented as
    one-way. This exists so the evaluation harness can score against the exact file the
    detector was trained on rather than a re-derived one.

    Images with no annotations are kept. They are the frames a false positive can appear
    in, and dropping them would remove most of the negative evidence.
    """
    if not path.is_file():
        raise ConfigError(f"COCO annotation file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    for key in ("images", "annotations", "categories"):
        if key not in raw:
            raise ConfigError(f"{path} has no {key!r} key; it is not a COCO detection file.")

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in raw["annotations"]:
        by_image[int(ann["image_id"])].append(ann)

    images: list[CocoImage] = []
    for entry in raw["images"]:
        image_id = int(entry["id"])
        anns = by_image.get(image_id, [])
        # COCO bbox is [x, y, w, h]; everything downstream of this line is xyxy.
        boxes = (
            xywh_to_xyxy(np.asarray([a["bbox"] for a in anns], dtype=np.float64))
            if anns
            else np.empty((0, 4), dtype=np.float64)
        )
        images.append(
            CocoImage(
                image_id=image_id,
                file_name=str(entry["file_name"]),
                width=int(entry["width"]),
                height=int(entry["height"]),
                boxes_xyxy=boxes,
                category_ids=np.asarray([int(a["category_id"]) for a in anns], dtype=np.int64),
                source_video=entry.get("source_video"),
            )
        )

    return CocoSplit(
        path=path,
        images=images,
        categories={int(c["id"]): str(c["name"]) for c in raw["categories"]},
    )
