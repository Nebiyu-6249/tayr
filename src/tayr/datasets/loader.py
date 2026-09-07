"""Load a directory of native annotation files into canonical form.

Directory conventions per format:

  dvb      one `*.txt` per video, filename stem is the video id.
  antiuav  one subdirectory per sequence, each containing `IR_label.json`;
           the subdirectory name is the video id. This matches the layout the
           benchmark's own toolkit globs for
           `[VERIFIED: antiuav410.py line 38, glob '*/IR_label.json']`.
  voc      one Pascal VOC XML per image in `xml/`, beside the image in `img/`.
           This is the DUT Anti-UAV **detection** subset's layout. The directory
           passed in is one split - `<root>/train`, `<root>/val`, `<root>/test`.

The `voc` loader carries findings the canonical form cannot hold - which index base the
coordinates were read under, how many annotations point at a missing image - through
`DatasetAnnotation.notes`, so `tayr dataset census` prints them next to the counts they
explain.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tayr.datasets.converters.anti_uav import parse_anti_uav_json
from tayr.datasets.converters.drone_vs_bird import parse_drone_vs_bird
from tayr.datasets.converters.voc import (
    VocAnnotation,
    VocIndexBase,
    gather_index_base_evidence,
    map_class,
    parse_voc_xml,
    voc_to_video,
)
from tayr.datasets.schema import DatasetAnnotation, ObjectClass, VideoAnnotation
from tayr.errors import ConfigError

SUPPORTED_FORMATS = ("dvb", "antiuav", "voc")

#: Where a VOC split keeps its annotations and its images.
VOC_XML_DIR = "xml"
VOC_IMG_DIR = "img"

# Licence and redistribution rights per format, from Phase 0 verification. These
# defaults are conservative: a dataset is assumed non-redistributable unless its
# licence was confirmed to permit it.
_LICENCE: dict[str, tuple[str, bool]] = {
    # Data usage agreement, research use only, no redistribution rights granted.
    "dvb": ("WOSDETC data usage agreement (no redistribution)", False),
    # MIT, but we still never place dataset-derived files in the public repo.
    "antiuav": ("MIT", False),
    # `voc` is a container format, not a dataset: DUT Anti-UAV is the one this was
    # written against, but any VOC-XML dataset parses. The licence therefore cannot be
    # inferred from the format and is left UNKNOWN for the config to state.
    "voc": ("UNKNOWN - set it from the dataset's own licence", False),
}


def load_native_directory(directory: Path, *, fmt: str, name: str, split: str) -> DatasetAnnotation:
    """Parse every annotation file under `directory` in the named native format."""
    if fmt == "dut":
        # The name a person reaches for first, and it is ambiguous: DUT ships two
        # subsets with different shapes and only one of them has anything to parse.
        raise ConfigError(
            "there is no 'dut' format, because DUT Anti-UAV ships two subsets that are "
            "not the same shape. Its DETECTION subset is Pascal VOC XML - use "
            "--format voc, pointing --dataset at one split (<root>/train). Its TRACKING "
            "subset ships one first-frame box per video and no per-frame ground truth, "
            "so there is no track annotation to convert; see docs/RESEARCH.md 14.4 and "
            "converters/dut_anti_uav.py."
        )
    if fmt not in SUPPORTED_FORMATS:
        raise ConfigError(
            f"unsupported format {fmt!r}; expected one of {', '.join(SUPPORTED_FORMATS)}."
        )
    if not directory.is_dir():
        raise ConfigError(f"not a directory: {directory}")

    if fmt == "voc":
        return load_voc_split(directory, name=name, split=split)

    videos: list[VideoAnnotation] = []

    if fmt == "dvb":
        paths = sorted(directory.glob("*.txt"))
        if not paths:
            raise ConfigError(f"no *.txt annotation files found under {directory}")
        for path in paths:
            videos.append(
                parse_drone_vs_bird(path.read_text(encoding="utf-8"), source_video=path.stem)
            )
    else:
        paths = sorted(directory.glob("*/IR_label.json"))
        if not paths:
            raise ConfigError(f"no */IR_label.json files found under {directory}")
        for track_id, path in enumerate(paths):
            video, _report = parse_anti_uav_json(
                path.read_text(encoding="utf-8"),
                source_video=path.parent.name,
                track_id=track_id,
            )
            videos.append(video)

    licence, redistributable = _LICENCE[fmt]
    return DatasetAnnotation(
        name=name,
        split=split,
        videos=tuple(videos),
        licence=licence,
        redistributable=redistributable,
    )


def read_voc_annotations(directory: Path) -> list[tuple[Path, VocAnnotation]]:
    """Parse every `xml/*.xml` under one split directory, in filename order."""
    xml_dir = directory / VOC_XML_DIR
    if not xml_dir.is_dir():
        raise ConfigError(
            f"no {VOC_XML_DIR}/ subdirectory under {directory}. The VOC layout this "
            f"reads is <split>/{VOC_XML_DIR}/*.xml beside <split>/{VOC_IMG_DIR}/*.jpg, "
            "and the directory passed here is one split."
        )
    paths = sorted(xml_dir.glob("*.xml"))
    if not paths:
        raise ConfigError(f"no *.xml annotation files found under {xml_dir}")
    return [
        (path, parse_voc_xml(path.read_text(encoding="utf-8"), path=str(path))) for path in paths
    ]


def load_voc_split(
    directory: Path,
    *,
    name: str,
    split: str,
    index_base: VocIndexBase = VocIndexBase.ZERO,
) -> DatasetAnnotation:
    """Load one Pascal VOC split into canonical form.

    Each image becomes its own single-frame `VideoAnnotation`. That is not a modelling
    convenience: a detection subset has no temporal structure, and the alternative
    encodings both assert something false. One video holding every image would claim a
    sequence that does not exist; grouping by filename prefix would invent the video
    identity this dataset does not ship.

    `file_name` is written as `img/<filename>` so a COCO file placed in the split
    directory resolves against the images without moving them - torchvision's
    `CocoDetection` opens `os.path.join(root, file_name)`
    `[VERIFIED: torchvision/datasets/coco.py _load_image, and rfdetr's CocoDetection
    subclasses it without overriding that method]`.
    """
    parsed = read_voc_annotations(directory)
    annotations = [annotation for _, annotation in parsed]

    videos: list[VideoAnnotation] = []
    for path, annotation in parsed:
        videos.append(
            voc_to_video(
                annotation,
                source_video=path.stem,
                index_base=index_base,
                image_prefix=f"{VOC_IMG_DIR}/",
            )
        )

    licence, redistributable = _LICENCE["voc"]
    return DatasetAnnotation(
        name=name,
        split=split,
        videos=tuple(videos),
        licence=licence,
        redistributable=redistributable,
        notes=tuple(_voc_notes(directory, annotations, index_base=index_base)),
    )


def _voc_notes(
    directory: Path, annotations: list[VocAnnotation], *, index_base: VocIndexBase
) -> list[str]:
    """Everything the native files said that the canonical form cannot carry."""
    notes = [
        f"COORDINATES READ AS {index_base.value.upper()}-BASED. "
        f"{gather_index_base_evidence(annotations).verdict}"
    ]

    missing = [
        annotation.filename
        for annotation in annotations
        if not (directory / VOC_IMG_DIR / annotation.filename).is_file()
    ]
    if missing:
        shown = ", ".join(missing[:5])
        notes.append(
            f"{len(missing)} of {len(annotations)} annotation(s) name an image that is "
            f"not in {directory / VOC_IMG_DIR} (first: {shown}). Counting them is fine; "
            "training on them is not, and `tayr dataset prepare` refuses to."
        )

    flags = Counter(
        {
            "difficult": sum(1 for a in annotations for o in a.objects if o.difficult),
            "truncated": sum(1 for a in annotations for o in a.objects if o.truncated),
        }
    )
    if flags["difficult"]:
        notes.append(
            f"{flags['difficult']} object(s) carry VOC's `difficult` flag. Tayr converts "
            "them as ordinary positives - the flag is a VOC evaluation convention with no "
            "equivalent in this project's protocol, which handles exclusions through "
            "pixels-on-target ranges instead. Saying so because it affects recall."
        )
    if flags["truncated"]:
        notes.append(f"{flags['truncated']} object(s) carry VOC's `truncated` flag; also kept.")

    unmapped = Counter(
        o.name for a in annotations for o in a.objects if map_class(o.name) is ObjectClass.UNKNOWN
    )
    if unmapped:
        notes.append(
            f"{sum(unmapped.values())} object(s) have a class name this converter does "
            f"not recognise and were labelled UNKNOWN rather than guessed: "
            f"{dict(unmapped.most_common(5))}. Add them to voc.CLASS_MAP if they are real."
        )
    return notes
