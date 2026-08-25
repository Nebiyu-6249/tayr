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

from typing import Any

from tayr.datasets.schema import DatasetAnnotation, ObjectClass

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
                    "file_name": f"{video.source_video}/{frame.frame_index:06d}.jpg",
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
