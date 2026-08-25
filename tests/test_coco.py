"""COCO export tests. The xywh/xyxy boundary is the classic place to lose a box."""

from __future__ import annotations

from tayr.datasets.coco import CATEGORY_IDS, to_coco
from tayr.datasets.converters.drone_vs_bird import parse_drone_vs_bird
from tayr.datasets.schema import DatasetAnnotation, ObjectClass


def _dataset(text: str) -> DatasetAnnotation:
    return DatasetAnnotation(
        name="t",
        split="train",
        videos=(parse_drone_vs_bird(text, source_video="v1"),),
        licence="test",
        redistributable=False,
    )


class TestCocoExport:
    def test_bbox_is_xywh_not_xyxy(self) -> None:
        """Canonical xyxy(100,200,130,240) must export as COCO [100,200,30,40]."""
        coco = to_coco(_dataset("0 1 100 200 30 40 drone\n"))
        assert coco["annotations"][0]["bbox"] == [100.0, 200.0, 30.0, 40.0]
        assert coco["annotations"][0]["area"] == 1200.0

    def test_empty_frames_are_exported_as_images(self) -> None:
        """Negative examples are what false-alarm rate is measured against."""
        coco = to_coco(_dataset("0 1 10 10 5 5 drone\n1 0\n2 0\n"))
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) == 1

    def test_source_video_survives_export(self) -> None:
        """Grouped-by-video splitting is impossible without this."""
        coco = to_coco(_dataset("0 1 10 10 5 5 drone\n"))
        assert coco["images"][0]["source_video"] == "v1"
        assert coco["images"][0]["frame_index"] == 0

    def test_category_ids_are_one_based(self) -> None:
        assert min(CATEGORY_IDS.values()) == 1
        assert CATEGORY_IDS[ObjectClass.DRONE] == 1
        coco = to_coco(_dataset("0 1 10 10 5 5 bird\n"))
        assert coco["annotations"][0]["category_id"] == CATEGORY_IDS[ObjectClass.BIRD]

    def test_ids_are_unique_and_link_correctly(self) -> None:
        coco = to_coco(_dataset("0 2 10 10 5 5 drone 40 40 6 6 bird\n1 1 11 11 5 5 drone\n"))
        image_ids = [i["id"] for i in coco["images"]]
        assert len(set(image_ids)) == len(image_ids)
        ann_ids = [a["id"] for a in coco["annotations"]]
        assert len(set(ann_ids)) == len(ann_ids)
        assert {a["image_id"] for a in coco["annotations"]} <= set(image_ids)
        assert sum(1 for a in coco["annotations"] if a["image_id"] == image_ids[0]) == 2

    def test_licence_metadata_travels_with_the_export(self) -> None:
        coco = to_coco(_dataset("0 1 10 10 5 5 drone\n"))
        assert coco["info"]["redistributable"] is False
        assert coco["info"]["licence"] == "test"
