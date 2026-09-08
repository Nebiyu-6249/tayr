"""Pascal VOC conversion.

The round trip is here because the skill demands it, but round trips are weak on their
own: native -> canonical -> native passes even when both directions share the same sign
error. So every geometric claim is also asserted against a hand-computed absolute value,
worked from the sample annotation in the DUT Anti-UAV detection subset.

That sample is 34x13 under the 1-based reading Tayr defaults to - 21.0 pixels on target -
and 33x12 under the 0-based one, 19.9. Both land in the same bucket, but on an 8px target
the same one-pixel shift is over 10% of the box, which is why the choice is measured
rather than inherited. The evidence is in docs/RESEARCH.md 14.6.
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
    DEFAULT_INDEX_BASE,
    MAX_XML_BYTES,
    VocAnnotation,
    VocIndexBase,
    VocObject,
    gather_index_base_evidence,
    map_class,
    parse_index_base,
    parse_voc_xml,
    video_to_voc,
    voc_to_video,
    write_voc_xml,
)
from tayr.datasets.loader import load_native_directory, load_voc_split
from tayr.datasets.prepare import PreparedDataset, prepare_detector_dataset
from tayr.datasets.schema import (
    BoxAnnotation,
    DatasetAnnotation,
    ObjectClass,
    TrackIdSource,
)
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


def only_box(document: str, *, index_base: VocIndexBase = DEFAULT_INDEX_BASE) -> BoxAnnotation:
    """Convert a one-box document, asserting nothing was rejected."""
    video, rejected = voc_to_video(parse_voc_xml(document), source_video="x", index_base=index_base)
    assert not rejected, rejected
    return video.frames[0].objects[0]


class TestTheDefault:
    def test_the_default_is_one_based(self) -> None:
        """Changed from ZERO after measurement on the real dataset; see RESEARCH.md 14.6."""
        assert DEFAULT_INDEX_BASE is VocIndexBase.ONE

    @pytest.mark.parametrize(
        ("text", "expected"), [("one", VocIndexBase.ONE), ("ZERO ", VocIndexBase.ZERO)]
    )
    def test_the_cli_string_parses(self, text: str, expected: VocIndexBase) -> None:
        assert parse_index_base(text) is expected

    def test_an_unknown_index_base_names_the_valid_ones(self) -> None:
        with pytest.raises(ConfigError, match="not recognised"):
            parse_index_base("half")

    def test_the_default_is_what_an_unspecified_conversion_uses(self) -> None:
        explicit = only_box(SAMPLE, index_base=VocIndexBase.ONE)
        implicit = only_box(SAMPLE)
        assert (implicit.x1, implicit.y1, implicit.x2, implicit.y2) == (
            explicit.x1,
            explicit.y1,
            explicit.x2,
            explicit.y2,
        )


class TestAbsoluteValues:
    """Hand-computed, because a round trip alone cannot catch a shared sign error."""

    def test_the_sample_box_converts_to_known_coordinates(self) -> None:
        box = only_box(SAMPLE)
        assert (box.x1, box.y1, box.x2, box.y2) == (868.0, 241.0, 902.0, 254.0)
        assert box.x2 - box.x1 == 34.0
        assert box.y2 - box.y1 == 13.0
        assert box.label is ObjectClass.DRONE

    def test_the_sample_box_pixels_on_target(self) -> None:
        arr = only_box(SAMPLE).as_array()
        assert float(pixels_on_target(arr)) == pytest.approx(math.sqrt(34 * 13))
        assert str(size_bucket(arr).item()) == SizeBucket.MEDIUM.value

    def test_coco_bbox_is_xywh_not_xyxy(self, tmp_path: Path) -> None:
        """The conversion the skill names as where the bugs live."""
        (tmp_path / "xml").mkdir()
        (tmp_path / "img").mkdir()
        (tmp_path / "xml" / "02425.xml").write_text(SAMPLE)
        (tmp_path / "img" / "02425.jpg").write_bytes(b"x")
        coco = to_coco(load_voc_split(tmp_path, name="dut", split="val"))
        assert coco["annotations"][0]["bbox"] == [868.0, 241.0, 34.0, 13.0]
        assert coco["annotations"][0]["area"] == 34.0 * 13.0

    def test_the_zero_based_reading_is_a_pixel_narrower(self) -> None:
        """The alternative, so the difference the default makes stays visible."""
        box = only_box(SAMPLE, index_base=VocIndexBase.ZERO)
        assert (box.x1, box.y1, box.x2, box.y2) == (869.0, 242.0, 902.0, 254.0)
        assert (box.x2 - box.x1, box.y2 - box.y1) == (33.0, 12.0)
        assert float(pixels_on_target(box.as_array())) == pytest.approx(math.sqrt(33 * 12))

    def test_the_two_readings_differ_by_more_than_a_tenth_on_a_small_box(self) -> None:
        """Why this is a decision and not a rounding detail."""
        document = xml_document("small.jpg", [(100, 100, 107, 107)])
        one = only_box(document, index_base=VocIndexBase.ONE)
        zero = only_box(document, index_base=VocIndexBase.ZERO)
        one_px = float(pixels_on_target(one.as_array()))
        zero_px = float(pixels_on_target(zero.as_array()))
        assert one_px == pytest.approx(8.0)
        assert zero_px == pytest.approx(7.0)
        assert (one_px - zero_px) / zero_px > 0.10

    def test_a_four_pixel_box_survives_conversion(self) -> None:
        """The bug this project must not have: a tiny box arriving with zero extent."""
        box = only_box(xml_document("tiny.jpg", [(100, 100, 103, 103)]))
        assert (box.x2 - box.x1, box.y2 - box.y1) == (4.0, 4.0)
        assert str(size_bucket(box.as_array()).item()) == SizeBucket.TINY.value


class TestRoundTrip:
    @pytest.mark.parametrize("index_base", list(VocIndexBase))
    def test_native_to_canonical_to_native_preserves_geometry(
        self, index_base: VocIndexBase
    ) -> None:
        original = parse_voc_xml(SAMPLE)
        video, rejected = voc_to_video(original, source_video="02425", index_base=index_base)
        assert not rejected
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
        video, rejected = voc_to_video(original, source_video="multi")
        assert not rejected
        assert len(video.frames[0].objects) == 2
        back = video_to_voc(video, class_names={ObjectClass.DRONE: "UAV"})
        assert [(o.xmin, o.ymin, o.xmax, o.ymax) for o in back.objects] == [
            (10, 20, 30, 40),
            (100, 200, 140, 260),
        ]

    def test_the_dropped_fields_are_dropped_deliberately(self) -> None:
        """truncated/difficult/pose/depth do not survive, and that is the design."""
        document = xml_document("d.jpg", [(10, 20, 30, 40)], difficult=1)
        video, _ = voc_to_video(parse_voc_xml(document), source_video="d")
        back = video_to_voc(video, class_names={ObjectClass.DRONE: "UAV"})
        assert back.objects[0].difficult == 0
        # Geometry is what must not be lost, and it is not.
        assert (back.objects[0].xmin, back.objects[0].ymax) == (10, 40)


class TestEmptyAndDegenerate:
    def test_a_file_with_no_object_is_a_frame_with_no_boxes(self) -> None:
        """VOC says 'nothing here' by carrying no <object>. That is a negative, not a gap."""
        video, rejected = voc_to_video(
            parse_voc_xml(xml_document("empty.jpg", [])), source_video="empty"
        )
        assert not rejected
        assert video.n_boxes == 0
        assert video.frames[0].is_empty
        assert video.n_empty_frames == 1

    def test_empty_files_are_kept_through_to_coco(self, tmp_path: Path) -> None:
        """Dropping them shrinks the false-alarm denominator and inflates the headline."""
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "b": []})
        coco = to_coco(load_voc_split(tmp_path / "train", name="dut", split="train"))
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 1

    def test_a_flat_box_is_one_pixel_under_the_default_and_rejected_under_zero(self) -> None:
        """This is the real 00991.jpg: ymin == ymax == 443.

        Zero height under the 0-based reading, one pixel under the 1-based one. It is the
        most direct piece of evidence in RESEARCH.md 14.6, and the case that used to
        abort a whole census run.
        """
        document = xml_document("00991.jpg", [(869, 443, 902, 443)])

        video, rejected = voc_to_video(parse_voc_xml(document), source_video="00991")
        assert not rejected
        assert video.frames[0].objects[0].y2 - video.frames[0].objects[0].y1 == 1.0

        video, rejected = voc_to_video(
            parse_voc_xml(document), source_video="00991", index_base=VocIndexBase.ZERO
        )
        assert video.n_boxes == 0
        assert len(rejected) == 1
        assert "non-positive extent" in rejected[0].reason
        assert (rejected[0].xmin, rejected[0].ymin) == (869, 443)

    def test_a_rejected_box_is_never_repaired_only_recorded(self) -> None:
        """Clamping would train a detector on a lie that no downstream number reveals."""
        document = xml_document("bad.jpg", [(100, 140, 100, 100)])
        video, rejected = voc_to_video(parse_voc_xml(document), source_video="bad")
        assert video.n_boxes == 0
        assert len(rejected) == 1
        assert rejected[0].index_base == "one"
        assert "bad.jpg" in rejected[0].render()

    def test_one_bad_box_does_not_cost_the_others_in_the_same_file(self) -> None:
        document = xml_document("mixed.jpg", [(10, 20, 30, 40), (50, 50, 50, 50)])
        video, rejected = voc_to_video(
            parse_voc_xml(document), source_video="mixed", index_base=VocIndexBase.ZERO
        )
        assert video.n_boxes == 1
        assert len(rejected) == 1

    def test_a_negative_origin_is_rejected_not_emitted(self) -> None:
        """xmin=0 under the 1-based reading becomes -1: a box starting outside the image."""
        document = xml_document("neg.jpg", [(0, 10, 30, 40)])
        video, rejected = voc_to_video(parse_voc_xml(document), source_video="neg")
        assert video.n_boxes == 0
        assert "outside the image" in rejected[0].reason

    def test_a_box_past_the_frame_edge_is_rejected(self) -> None:
        document = xml_document("over.jpg", [(10, 10, 5000, 40)], width=1920, height=1080)
        video, rejected = voc_to_video(parse_voc_xml(document), source_video="over")
        assert video.n_boxes == 0
        assert "past the 1920x1080 frame" in rejected[0].reason


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
    def test_a_zero_minimum_rules_out_the_one_based_reading(self) -> None:
        """A 1-based coordinate cannot be zero, so one zero settles it."""
        annotations = [
            parse_voc_xml(xml_document("a.jpg", [(0, 40, 20, 60)])),
            parse_voc_xml(xml_document("b.jpg", [(30, 40, 50, 60)])),
        ]
        evidence = gather_index_base_evidence(annotations)
        assert evidence.n_zero_minimums == 1
        assert evidence.rules_out_one_based
        assert "0-BASED, proven" in evidence.verdict_for(VocIndexBase.ZERO)

    def test_reading_as_one_based_against_a_zero_coordinate_is_called_out(self) -> None:
        """The default is ONE, so this is the case where the default is wrong."""
        evidence = gather_index_base_evidence(
            [parse_voc_xml(xml_document("a.jpg", [(0, 40, 20, 60)]))]
        )
        verdict = evidence.verdict_for(VocIndexBase.ONE)
        assert "outside the image" in verdict
        assert "index_base=zero" in verdict

    def test_a_flat_box_indicates_the_one_based_reading(self) -> None:
        """xmin == xmax is unusable under ZERO and one pixel under ONE."""
        evidence = gather_index_base_evidence(
            [parse_voc_xml(xml_document("00991.jpg", [(869, 443, 902, 443)]))]
        )
        assert evidence.unusable_under_zero_based
        assert "1-BASED consistent" in evidence.verdict_for(VocIndexBase.ONE)
        assert "1-BASED indicated" in evidence.verdict_for(VocIndexBase.ZERO)

    def test_both_signals_at_once_is_reported_as_contradictory(self) -> None:
        """No single reading makes the split valid, so some annotations are defective."""
        evidence = gather_index_base_evidence(
            [
                parse_voc_xml(xml_document("a.jpg", [(0, 40, 20, 60)])),
                parse_voc_xml(xml_document("b.jpg", [(50, 50, 50, 80)])),
            ]
        )
        assert evidence.is_contradictory
        for base in VocIndexBase:
            assert "CONTRADICTORY" in evidence.verdict_for(base)

    def test_no_signal_either_way_says_unproven_rather_than_guessing(self) -> None:
        evidence = gather_index_base_evidence([parse_voc_xml(SAMPLE)])
        assert evidence.n_zero_minimums == 0
        assert not evidence.unusable_under_zero_based
        assert "UNPROVEN" in evidence.verdict_for(VocIndexBase.ONE)
        assert evidence.min_coordinate == 242

    def test_no_boxes_infers_nothing(self) -> None:
        evidence = gather_index_base_evidence([parse_voc_xml(xml_document("e.jpg", []))])
        assert "nothing to infer" in evidence.verdict_for(VocIndexBase.ONE)


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
        assert any("READ AS ONE-BASED" in note for note in dataset.notes)

    def test_an_overridden_index_base_is_recorded_too(self, tmp_path: Path) -> None:
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)]})
        dataset = load_voc_split(
            tmp_path / "train", name="dut", split="train", index_base=VocIndexBase.ZERO
        )
        assert any("READ AS ZERO-BASED" in note for note in dataset.notes)

    def test_a_split_survives_one_unconvertible_box(self, tmp_path: Path) -> None:
        """The 00991.jpg case: one bad box must not cost the other files.

        Aborting here is what made `tayr dataset census` on val report nothing at all
        rather than reporting the 2,599 frames it could read.
        """
        write_split(
            tmp_path,
            "train",
            {
                "good": [(10, 10, 30, 30)],
                "00991": [(869, 443, 902, 443)],
                "alsogood": [(40, 40, 60, 60)],
            },
        )
        dataset = load_voc_split(
            tmp_path / "train", name="dut", split="train", index_base=VocIndexBase.ZERO
        )
        assert dataset.n_frames == 3
        assert dataset.n_boxes == 2
        assert any("REJECTED, not repaired" in note for note in dataset.notes)
        assert any("00991.jpg" in note for note in dataset.notes)

    def test_rejections_reach_the_census_and_the_coco_info_block(self, tmp_path: Path) -> None:
        """A COCO file quietly a box short reads as poor recall and nothing says otherwise."""
        write_split(tmp_path, "train", {"a": [(10, 10, 30, 30)], "00991": [(869, 443, 902, 443)]})
        dataset = load_voc_split(
            tmp_path / "train", name="dut", split="train", index_base=VocIndexBase.ZERO
        )
        assert "REJECTED" in take_census(dataset).render()
        notes = to_coco(dataset)["info"]["tayr_notes"]
        assert any("REJECTED" in note for note in notes)
        assert any("ZERO-BASED" in note for note in notes)

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
