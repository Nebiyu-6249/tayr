"""Pascal VOC conversion.

The round trip is here because the skill demands it, but round trips are weak on their
own: native -> canonical -> native passes even when both directions share the same sign
error. So every geometric claim is also asserted against a hand-computed absolute value,
worked from the sample annotation in the DUT Anti-UAV detection subset.

That sample is a 33x12 box, which is 19.9 pixels on target. The index-base choice moves
it to 21.0. Those land in the same bucket here, but on an 8px target the same one-pixel
shift is over 10% of the box, which is why the choice is explicit and tested rather than
inherited from whichever convention the parser happened to assume.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from tayr.datasets.census import take_census
from tayr.datasets.coco import to_coco
from tayr.datasets.converters.voc import (
    MAX_XML_BYTES,
    VocAnnotation,
    VocIndexBase,
    VocObject,
    gather_index_base_evidence,
    map_class,
    parse_voc_xml,
    video_to_voc,
    voc_to_video,
    write_voc_xml,
)
from tayr.datasets.loader import load_native_directory, load_voc_split
from tayr.datasets.prepare import PreparedDataset, prepare_detector_dataset
from tayr.datasets.schema import DatasetAnnotation, ObjectClass, TrackIdSource
from tayr.errors import ConfigError
from tayr.geometry import SizeBucket, pixels_on_target, size_bucket

#: The sample annotation from the DUT Anti-UAV detection subset, verbatim.
SAMPLE = """<annotation>
  <folder>val</folder>
  <filename>02425.jpg</filename>
  <size><width>1920</width><height>1080</height><depth>3</depth></size>
  <object>
    <name>UAV</name>
    <truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>869</xmin><ymin>242</ymin><xmax>902</xmax><ymax>254</ymax></bndbox>
  </object>
</annotation>"""


def xml_document(filename: str, boxes: list[tuple[int, int, int, int]], **kwargs: object) -> str:
    """Build a VOC document. `boxes` are (xmin, ymin, xmax, ymax)."""
    name = kwargs.get("name", "UAV")
    difficult = kwargs.get("difficult", 0)
    width = kwargs.get("width", 1920)
    height = kwargs.get("height", 1080)
    objects = "".join(
        f"<object><name>{name}</name><truncated>0</truncated>"
        f"<difficult>{difficult}</difficult><bndbox>"
        f"<xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
        f"</bndbox></object>"
        for a, b, c, d in boxes
    )
    return (
        f"<annotation><folder>train</folder><filename>{filename}</filename>"
        f"<size><width>{width}</width><height>{height}</height><depth>3</depth></size>"
        f"{objects}</annotation>"
    )


def write_split(
    root: Path,
    split: str,
    files: dict[str, list[tuple[int, int, int, int]]],
    *,
    images: bool = True,
) -> Path:
    """Lay out one split as <root>/<split>/{xml,img}."""
    directory = root / split
    (directory / "xml").mkdir(parents=True, exist_ok=True)
    (directory / "img").mkdir(parents=True, exist_ok=True)
    for stem, boxes in files.items():
        (directory / "xml" / f"{stem}.xml").write_text(xml_document(f"{stem}.jpg", boxes))
        if images:
            (directory / "img" / f"{stem}.jpg").write_bytes(b"not a real jpeg")
    return directory


class TestAbsoluteValues:
    """Hand-computed, because a round trip alone cannot catch a shared sign error."""

    def test_the_sample_box_converts_to_known_coordinates(self) -> None:
        video = voc_to_video(parse_voc_xml(SAMPLE), source_video="02425")
        box = video.frames[0].objects[0]
        assert (box.x1, box.y1, box.x2, box.y2) == (869.0, 242.0, 902.0, 254.0)
        assert box.x2 - box.x1 == 33.0
        assert box.y2 - box.y1 == 12.0
        assert box.label is ObjectClass.DRONE

    def test_the_sample_box_pixels_on_target(self) -> None:
        video = voc_to_video(parse_voc_xml(SAMPLE), source_video="02425")
        arr = video.frames[0].objects[0].as_array()
        assert float(pixels_on_target(arr)) == pytest.approx(math.sqrt(33 * 12))
        assert str(size_bucket(arr).item()) == SizeBucket.MEDIUM.value

    def test_coco_bbox_is_xywh_not_xyxy(self, tmp_path: Path) -> None:
        """The conversion the skill names as where the bugs live."""
        (tmp_path / "xml").mkdir()
        (tmp_path / "img").mkdir()
        (tmp_path / "xml" / "02425.xml").write_text(SAMPLE)
        (tmp_path / "img" / "02425.jpg").write_bytes(b"x")
        coco = to_coco(load_voc_split(tmp_path, name="dut", split="val"))
        assert coco["annotations"][0]["bbox"] == [869.0, 242.0, 33.0, 12.0]
        assert coco["annotations"][0]["area"] == 33.0 * 12.0

    def test_the_one_based_reading_shifts_origin_and_widens(self) -> None:
        """The alternative, so the difference the default avoids is visible."""
        video = voc_to_video(
            parse_voc_xml(SAMPLE), source_video="02425", index_base=VocIndexBase.ONE
        )
        box = video.frames[0].objects[0]
        assert (box.x1, box.y1, box.x2, box.y2) == (868.0, 241.0, 902.0, 254.0)
        assert (box.x2 - box.x1, box.y2 - box.y1) == (34.0, 13.0)
        assert float(pixels_on_target(box.as_array())) == pytest.approx(math.sqrt(34 * 13))

    def test_a_four_pixel_box_survives_conversion(self) -> None:
        """The bug this project must not have: a tiny box arriving with zero extent."""
        document = xml_document("tiny.jpg", [(100, 100, 104, 104)])
        box = voc_to_video(parse_voc_xml(document), source_video="tiny").frames[0].objects[0]
        assert (box.x2 - box.x1, box.y2 - box.y1) == (4.0, 4.0)
        assert str(size_bucket(box.as_array()).item()) == SizeBucket.TINY.value


class TestRoundTrip:
    @pytest.mark.parametrize("index_base", list(VocIndexBase))
    def test_native_to_canonical_to_native_preserves_geometry(
        self, index_base: VocIndexBase
    ) -> None:
        original = parse_voc_xml(SAMPLE)
        video = voc_to_video(original, source_video="02425", index_base=index_base)
        back = video_to_voc(
            video, index_base=index_base, class_names={ObjectClass.DRONE: "UAV"}, folder="val"
        )
        assert back.objects == original.objects
        assert (back.filename, back.width, back.height) == (
            original.filename,
            original.width,
            original.height,
        )

    def test_reserialised_xml_reparses_identically(self) -> None:
        original = parse_voc_xml(SAMPLE)
        assert parse_voc_xml(write_voc_xml(original)) == original

    def test_multiple_objects_round_trip_in_order(self) -> None:
        document = xml_document("multi.jpg", [(10, 20, 30, 40), (100, 200, 140, 260)])
        original = parse_voc_xml(document)
        video = voc_to_video(original, source_video="multi")
        assert len(video.frames[0].objects) == 2
        back = video_to_voc(video, class_names={ObjectClass.DRONE: "UAV"})
        assert [(o.xmin, o.ymin, o.xmax, o.ymax) for o in back.objects] == [
            (10, 20, 30, 40),
            (100, 200, 140, 260),
        ]

    def test_the_dropped_fields_are_dropped_deliberately(self) -> None:
        """truncated/difficult/pose/depth do not survive, and that is the design."""
        document = xml_document("d.jpg", [(10, 20, 30, 40)], difficult=1)
        video = voc_to_video(parse_voc_xml(document), source_video="d")
        back = video_to_voc(video, class_names={ObjectClass.DRONE: "UAV"})
        assert back.objects[0].difficult == 0
        # Geometry is what must not be lost, and it is not.
        assert (back.objects[0].xmin, back.objects[0].ymax) == (10, 40)


class TestEmptyAndDegenerate:
    def test_a_file_with_no_object_is_a_frame_with_no_boxes(self) -> None:
        """VOC says 'nothing here' by carrying no <object>. That is a negative, not a gap."""
        video = voc_to_video(parse_voc_xml(xml_document("empty.jpg", [])), source_video="empty")
        assert video.n_boxes == 0
        assert video.frames[0].is_empty
        assert video.n_empty_frames == 1

    def test_empty_files_are_kept_through_to_coco(self, tmp_path: Path) -> None:
        """Dropping them shrinks the false-alarm denominator and inflates the headline."""
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "b": []})
        coco = to_coco(load_voc_split(tmp_path / "train", name="dut", split="train"))
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 1

    def test_a_zero_extent_box_raises_rather_than_being_clamped(self) -> None:
        document = xml_document("bad.jpg", [(100, 100, 100, 140)])
        with pytest.raises(ConfigError, match="non-positive extent"):
            voc_to_video(parse_voc_xml(document), source_video="bad")

    def test_a_one_pixel_box_is_degenerate_under_the_zero_based_reading(self) -> None:
        """Honest consequence of the default, asserted rather than discovered later.

        `xmin == xmax` has zero width when read as 0-based-exclusive. It is one pixel
        wide when read as the devkit does, which is exactly the kind of dataset that
        needs `index_base=ONE`.
        """
        document = xml_document("px.jpg", [(50, 50, 50, 50)])
        with pytest.raises(ConfigError, match="non-positive extent"):
            voc_to_video(parse_voc_xml(document), source_video="px")
        one = voc_to_video(parse_voc_xml(document), source_video="px", index_base=VocIndexBase.ONE)
        assert one.frames[0].objects[0].x2 - one.frames[0].objects[0].x1 == 1.0


class TestClassMapping:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("UAV", ObjectClass.DRONE),
            ("uav", ObjectClass.DRONE),
            (" Drone ", ObjectClass.DRONE),
            ("bird", ObjectClass.BIRD),
            ("helicopter", ObjectClass.AIRCRAFT),
        ],
    )
    def test_known_tokens_map(self, token: str, expected: ObjectClass) -> None:
        assert map_class(token) is expected

    def test_an_unknown_token_becomes_unknown_never_drone(self) -> None:
        """Coercing an unrecognised class to DRONE would silently fabricate labels."""
        assert map_class("weather balloon") is ObjectClass.UNKNOWN


class TestParserHardening:
    def test_a_doctype_is_refused(self) -> None:
        """ElementTree expands internal entities; a DTD is the only place to declare one."""
        bomb = (
            '<?xml version="1.0"?>\n<!DOCTYPE lolz [<!ENTITY lol "lol">]>\n'
            "<annotation><filename>&lol;</filename></annotation>"
        )
        with pytest.raises(ConfigError, match="DOCTYPE"):
            parse_voc_xml(bomb)

    def test_an_oversized_document_is_refused_before_parsing(self) -> None:
        with pytest.raises(ConfigError, match="larger than"):
            parse_voc_xml("<annotation>" + "x" * MAX_XML_BYTES + "</annotation>")

    def test_malformed_xml_names_the_file(self) -> None:
        with pytest.raises(ConfigError, match="not well-formed"):
            parse_voc_xml("<annotation><filename>", path="broken.xml")

    def test_a_wrong_root_element_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="expected <annotation>"):
            parse_voc_xml("<dataset><filename>a.jpg</filename></dataset>")

    def test_a_missing_size_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="<size> is missing"):
            parse_voc_xml("<annotation><filename>a.jpg</filename></annotation>")

    def test_a_non_integer_coordinate_is_refused(self) -> None:
        document = (
            "<annotation><filename>a.jpg</filename>"
            "<size><width>10</width><height>10</height></size>"
            "<object><name>UAV</name><bndbox><xmin>a</xmin><ymin>1</ymin>"
            "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>"
        )
        with pytest.raises(ConfigError, match="not an integer"):
            parse_voc_xml(document)


class TestIndexBaseEvidence:
    def test_a_zero_minimum_proves_zero_based(self) -> None:
        """A 1-based coordinate cannot be zero, so one zero settles it."""
        annotations = [
            parse_voc_xml(xml_document("a.jpg", [(0, 40, 20, 60)])),
            parse_voc_xml(xml_document("b.jpg", [(30, 40, 50, 60)])),
        ]
        evidence = gather_index_base_evidence(annotations)
        assert evidence.n_zero_minimums == 1
        assert "0-BASED, proven" in evidence.verdict

    def test_no_zero_minimum_says_unproven_rather_than_guessing(self) -> None:
        annotations = [parse_voc_xml(SAMPLE)]
        evidence = gather_index_base_evidence(annotations)
        assert evidence.n_zero_minimums == 0
        assert "UNPROVEN" in evidence.verdict
        assert evidence.min_coordinate == 242

    def test_no_boxes_infers_nothing(self) -> None:
        evidence = gather_index_base_evidence([parse_voc_xml(xml_document("e.jpg", []))])
        assert "nothing to infer" in evidence.verdict


class TestLoader:
    def test_each_image_becomes_its_own_single_frame_group(self, tmp_path: Path) -> None:
        """A bag of stills has no temporal structure, and must not claim any."""
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "b": [(40, 40, 60, 60)]})
        dataset = load_voc_split(tmp_path / "train", name="dut", split="train")
        assert dataset.n_videos == 2
        assert dataset.n_frames == 2
        assert all(len(v.frames) == 1 for v in dataset.videos)
        assert all(v.track_id_source is TrackIdSource.NONE for v in dataset.videos)

    def test_file_names_point_at_the_image_directory(self, tmp_path: Path) -> None:
        """`img/<name>` resolves against the split directory the COCO file sits in."""
        write_split(tmp_path, "train", {"02425": [(10, 10, 30, 30)]})
        dataset = load_voc_split(tmp_path / "train", name="dut", split="train")
        assert dataset.videos[0].frames[0].file_name == "img/02425.jpg"

    def test_a_missing_xml_directory_names_the_expected_layout(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        with pytest.raises(ConfigError, match="no xml/ subdirectory"):
            load_voc_split(tmp_path / "train", name="dut", split="train")

    def test_an_empty_xml_directory_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "xml").mkdir(parents=True)
        with pytest.raises(ConfigError, match=re.escape("no *.xml annotation files")):
            load_voc_split(tmp_path / "train", name="dut", split="train")

    def test_the_index_base_used_is_recorded_in_the_notes(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)]})
        dataset = load_voc_split(tmp_path / "train", name="dut", split="train")
        assert any("READ AS ZERO-BASED" in note for note in dataset.notes)

    def test_missing_images_are_counted_in_the_notes(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)]}, images=False)
        dataset = load_voc_split(tmp_path / "train", name="dut", split="train")
        assert any("name an image that is not in" in note for note in dataset.notes)

    def test_difficult_objects_are_reported_not_silently_kept(self, tmp_path: Path) -> None:
        directory = tmp_path / "train"
        (directory / "xml").mkdir(parents=True)
        (directory / "img").mkdir(parents=True)
        (directory / "xml" / "a.xml").write_text(
            xml_document("a.jpg", [(10, 10, 30, 30)], difficult=1)
        )
        (directory / "img" / "a.jpg").write_bytes(b"x")
        dataset = load_voc_split(directory, name="dut", split="train")
        assert any("difficult" in note for note in dataset.notes)

    def test_unrecognised_class_names_are_reported(self, tmp_path: Path) -> None:
        directory = tmp_path / "train"
        (directory / "xml").mkdir(parents=True)
        (directory / "img").mkdir(parents=True)
        (directory / "xml" / "a.xml").write_text(
            xml_document("a.jpg", [(10, 10, 30, 30)], name="quadcopter")
        )
        (directory / "img" / "a.jpg").write_bytes(b"x")
        dataset = load_voc_split(directory, name="dut", split="train")
        assert any("does not recognise" in note for note in dataset.notes)

    def test_voc_is_reachable_through_the_generic_loader(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)]})
        dataset = load_native_directory(tmp_path / "train", fmt="voc", name="dut", split="train")
        assert dataset.n_boxes == 1


class TestCensus:
    def test_zero_tracks_is_stated_plainly(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "b": [(40, 40, 60, 60)]})
        census = take_census(load_voc_split(tmp_path / "train", name="dut", split="train"))
        assert census.n_tracks == 0
        assert any("ZERO tracks" in w for w in census.warnings)

    def test_the_absence_of_video_grouping_is_stated(self, tmp_path: Path) -> None:
        """Not just 'no tracks' - the grouped-split guarantee has nothing to group on."""
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "b": [(40, 40, 60, 60)]})
        rendered = take_census(
            load_voc_split(tmp_path / "train", name="dut", split="train")
        ).render()
        assert "NO VIDEO GROUPING" in rendered
        assert "upper bound" in rendered

    def test_loader_notes_reach_the_rendered_census(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(0, 10, 30, 30)]})
        rendered = take_census(
            load_voc_split(tmp_path / "train", name="dut", split="train")
        ).render()
        assert "0-BASED, proven" in rendered

    def test_no_track_is_inferred_from_filename_order(self, tmp_path: Path) -> None:
        """Consecutively numbered stills are not a track and must not become one."""
        write_split(
            tmp_path,
            "train",
            {f"{i:05d}": [(10 + i, 10, 30 + i, 30)] for i in range(5)},
        )
        census = take_census(load_voc_split(tmp_path / "train", name="dut", split="train"))
        assert census.n_tracks == 0
        assert census.tracks_per_class.get("drone", 0) == 0


class TestPrepare:
    def prepared(self, tmp_path: Path) -> tuple[Path, PreparedDataset]:
        source = tmp_path / "src"
        for split in ("train", "val", "test"):
            write_split(source, split, {f"{split}0": [(10, 10, 30, 30)]})
        destination = tmp_path / "out"
        return destination, prepare_detector_dataset(
            source, destination_root=destination, name="dut"
        )

    def test_val_is_written_as_valid(self, tmp_path: Path) -> None:
        """RF-DETR's roboflow loader reads `valid/`, and DUT ships `val/`."""
        destination, _ = self.prepared(tmp_path)
        assert (destination / "valid" / "_annotations.coco.json").is_file()
        assert not (destination / "val").exists()

    def test_every_split_gets_the_filename_rf_detr_globs_for(self, tmp_path: Path) -> None:
        destination, _ = self.prepared(tmp_path)
        for split in ("train", "valid", "test"):
            assert (destination / split / "_annotations.coco.json").is_file()

    def test_images_are_linked_not_copied(self, tmp_path: Path) -> None:
        destination, _ = self.prepared(tmp_path)
        assert (destination / "train" / "img").is_symlink()
        assert (destination / "train" / "img" / "train0.jpg").is_file()

    def test_a_second_prepare_into_the_same_tree_refuses(self, tmp_path: Path) -> None:
        destination, _ = self.prepared(tmp_path)
        with pytest.raises(ConfigError, match="already exists"):
            prepare_detector_dataset(tmp_path / "src", destination_root=destination, name="dut")

    def test_annotations_naming_missing_images_are_refused(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        write_split(source, "train", {"a": [(10, 10, 30, 30)]}, images=False)
        with pytest.raises(ConfigError, match="point at images that are not there"):
            prepare_detector_dataset(source, destination_root=tmp_path / "out", name="dut")

    def test_a_missing_train_split_is_fatal(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        write_split(source, "test", {"a": [(10, 10, 30, 30)]})
        with pytest.raises(ConfigError, match="no train split"):
            prepare_detector_dataset(source, destination_root=tmp_path / "out", name="dut")

    def test_a_missing_test_split_is_only_a_warning(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        write_split(source, "train", {"a": [(10, 10, 30, 30)]})
        prepared = prepare_detector_dataset(source, destination_root=tmp_path / "out", name="dut")
        assert any("no val/ split" in w for w in prepared.warnings)
        assert any("no test/ split" in w for w in prepared.warnings)

    def test_writing_into_a_git_work_tree_is_refused(self, tmp_path: Path) -> None:
        """Derived annotations are text, so licence-guard would not catch a commit."""
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=30)
        source = tmp_path / "src"
        write_split(source, "train", {"a": [(10, 10, 30, 30)]})
        with pytest.raises(ConfigError, match="non-redistributable"):
            prepare_detector_dataset(source, destination_root=repo / "data", name="dut")

    def test_the_coco_file_is_valid_and_paths_resolve(self, tmp_path: Path) -> None:
        destination, _ = self.prepared(tmp_path)
        payload = json.loads(
            (destination / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        for image in payload["images"]:
            assert (destination / "train" / image["file_name"]).is_file()


class TestPreviewSelection:
    """Selection is tested without OpenCV; rendering needs the cv extra."""

    def dataset(self, tmp_path: Path) -> DatasetAnnotation:
        write_split(
            tmp_path,
            "train",
            {
                "small": [(100, 100, 105, 105)],
                "large": [(100, 100, 300, 300)],
                "multi": [(10, 10, 40, 40), (200, 200, 240, 240)],
                "empty": [],
                "plain": [(500, 500, 540, 540)],
            },
        )
        return load_voc_split(tmp_path / "train", name="dut", split="train")

    def test_the_interesting_frames_are_chosen_not_the_first_n(self, tmp_path: Path) -> None:
        from tayr.datasets.preview import choose_samples

        reasons = {s.video.source_video: s.reason for s in choose_samples(self.dataset(tmp_path))}
        assert reasons["small"] == "smallest box in the split"
        assert reasons["large"] == "largest box in the split"
        assert "empty" in reasons
        assert "no objects" in reasons["empty"]

    def test_selection_is_deterministic(self, tmp_path: Path) -> None:
        from tayr.datasets.preview import choose_samples

        dataset = self.dataset(tmp_path)
        first = [s.video.source_video for s in choose_samples(dataset, limit=4)]
        second = [s.video.source_video for s in choose_samples(dataset, limit=4)]
        assert first == second

    def test_the_limit_is_honoured(self, tmp_path: Path) -> None:
        from tayr.datasets.preview import choose_samples

        assert len(choose_samples(self.dataset(tmp_path), limit=2)) == 2

    def test_a_zero_limit_is_refused(self, tmp_path: Path) -> None:
        from tayr.datasets.preview import choose_samples

        with pytest.raises(ConfigError, match="at least 1"):
            choose_samples(self.dataset(tmp_path), limit=0)


def test_a_voc_annotation_reports_emptiness() -> None:
    assert VocAnnotation(filename="a.jpg", width=10, height=10).is_empty
    assert not VocAnnotation(
        filename="a.jpg",
        width=10,
        height=10,
        objects=(VocObject(name="UAV", xmin=1, ymin=1, xmax=5, ymax=5),),
    ).is_empty
