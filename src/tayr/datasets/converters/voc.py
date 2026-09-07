"""Pascal VOC XML annotation converter.

Written against the DUT Anti-UAV **detection** subset, whose layout is one XML file per
image in `<split>/xml/` beside the image in `<split>/img/`. The XML is standard VOC:

    <annotation>
      <folder>val</folder>
      <filename>02425.jpg</filename>
      <size><width>1920</width><height>1080</height><depth>3</depth></size>
      <object>
        <name>UAV</name>
        <truncated>0</truncated><difficult>0</difficult>
        <bndbox><xmin>869</xmin><ymin>242</ymin><xmax>902</xmax><ymax>254</ymax></bndbox>
      </object>
    </annotation>

## The index-base decision, made deliberately

VOC gives integer pixel corners. Whether `xmin` counts from 0 or 1, and whether `xmax`
is inclusive, changes every box by one pixel in origin and one in extent. On a 12px-tall
target - the sample above - that is 8% of the box, and this project's whole subject is
targets that small. It is not a rounding detail.

**Tayr defaults to `VocIndexBase.ZERO`:** `x1 = xmin`, `x2 = xmax`, so `width =
xmax - xmin`, the box read as the array slice `img[ymin:ymax, xmin:xmax]`.

The reasoning, with what is and is not established:

- The original Pascal VOC **devkit** is widely described as 1-based with an inclusive
  `xmax`, which would mean `width = xmax - xmin + 1`. **`[UNKNOWN]` here** - the devkit
  documentation at `host.robots.ox.ac.uk` returns 403 through this environment's egress
  proxy, so it has not been read against a primary source and is not asserted.
- What *is* established: torchvision's `VOCDetection.parse_voc_xml` performs **no**
  coordinate arithmetic at all and hands back the XML values verbatim
  `[VERIFIED: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/voc.py,
  fetched this session - zero occurrences of any arithmetic on the bndbox values]`.
  The most widely used VOC loader therefore applies no shift, and neither do the
  annotation tools that write this format today.
- **Which convention DUT Anti-UAV itself used is `[UNKNOWN]`** and cannot be settled
  from the schema. It can be settled from the data, so this module measures rather than
  assumes: a single `xmin` or `ymin` of 0 anywhere in a split is *proof* the file is not
  1-based, because a 1-based coordinate cannot be zero. `IndexBaseEvidence` reports what
  was found and `tayr dataset census` prints it.

Pass `index_base=VocIndexBase.ONE` for a dataset that genuinely follows the devkit. The
choice is recorded in the conversion output, so a run can say which reading produced it.

## Security

XML parsing is an attack surface and these files arrive from a third-party download.
Two findings, both from testing the parser in this session rather than from recollection:

- `xml.etree.ElementTree` **does** expand internal entities, so a billion-laughs file
  would exhaust memory `[VERIFIED: a three-level entity bomb expanded to 3000 chars]`.
- It **refuses** external entities, so file disclosure via XXE is not available
  `[VERIFIED: `file:///etc/passwd` entity raised "undefined entity"]`.

Entity declarations can only appear in a DTD `[VERIFIED: the same bomb without a
DOCTYPE raised "undefined entity"]`, so rejecting any document carrying a DOCTYPE closes
the expansion vector completely, with no dependency added and no legitimate VOC file
refused. `parse_voc_xml` does that, plus a size cap.

LICENCE: DUT Anti-UAV is Apache-2.0 `[VERIFIED: repository sidebar, Phase 0]`. Never
commit the imagery or the converted annotation files regardless - see CLAUDE.md 4.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from xml.etree.ElementTree import Element

from tayr.datasets.schema import (
    BoxAnnotation,
    FrameAnnotation,
    ObjectClass,
    TrackIdSource,
    VideoAnnotation,
)
from tayr.errors import ConfigError

#: Refuse anything larger. A VOC annotation for one image is a few hundred bytes; a
#: megabyte of it is not a real annotation and should not be handed to a parser.
MAX_XML_BYTES = 1024 * 1024

#: Class tokens seen in anti-UAV VOC datasets, lowercased. Anything unrecognised becomes
#: UNKNOWN and is counted, never silently coerced to DRONE.
CLASS_MAP: dict[str, ObjectClass] = {
    "uav": ObjectClass.DRONE,
    "drone": ObjectClass.DRONE,
    "bird": ObjectClass.BIRD,
    "airplane": ObjectClass.AIRCRAFT,
    "aeroplane": ObjectClass.AIRCRAFT,
    "aircraft": ObjectClass.AIRCRAFT,
    "helicopter": ObjectClass.AIRCRAFT,
}


class VocIndexBase(StrEnum):
    """How to read VOC's integer pixel corners. See the module docstring."""

    ZERO = "zero"
    """`x1 = xmin`, `x2 = xmax`; width is `xmax - xmin`. Tayr's default."""

    ONE = "one"
    """Devkit reading: 1-based with an inclusive max, so `x1 = xmin - 1` and width is
    `xmax - xmin + 1`."""


@dataclass(frozen=True, slots=True)
class VocObject:
    """One `<object>` block, verbatim. Integers, exactly as they appear in the file."""

    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    truncated: int = 0
    difficult: int = 0
    pose: str | None = None


@dataclass(frozen=True, slots=True)
class VocAnnotation:
    """One VOC XML file, verbatim. Native-to-native round-trips through this losslessly."""

    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...] = ()
    folder: str | None = None
    depth: int = 3

    @property
    def is_empty(self) -> bool:
        """VOC represents 'no target here' by simply carrying no `<object>` block."""
        return not self.objects


@dataclass(frozen=True, slots=True)
class IndexBaseEvidence:
    """What the data says about whether it is 0-based or 1-based.

    A zero minimum coordinate is proof of 0-based: a 1-based coordinate cannot be zero.
    Its absence proves nothing either way, which is why `verdict` says so rather than
    picking.
    """

    n_boxes: int
    min_coordinate: int | None
    n_zero_minimums: int
    n_max_at_edge: int
    """Boxes whose `xmax == width` or `ymax == height`. Legal under both readings, so
    this is context, not evidence."""

    @property
    def verdict(self) -> str:
        if self.n_boxes == 0:
            return "no boxes, nothing to infer"
        if self.n_zero_minimums:
            return (
                f"0-BASED, proven: {self.n_zero_minimums} box(es) have a zero minimum "
                "coordinate, which a 1-based annotation cannot produce"
            )
        return (
            f"UNPROVEN: no zero minimum coordinate in {self.n_boxes} box(es) "
            f"(smallest is {self.min_coordinate}). Consistent with either reading; "
            "the default 0-based conversion is being used. Check the rendered previews."
        )


def _text(node: Element, tag: str, *, path: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        raise ConfigError(f"{path}: <{tag}> is missing or empty")
    return child.text.strip()


def _int(node: Element, tag: str, *, path: str) -> int:
    raw = _text(node, tag, path=path)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{path}: <{tag}> is {raw!r}, which is not an integer") from exc


def _optional_int(node: Element, tag: str, default: int) -> int:
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return int(child.text.strip())
    except ValueError:
        return default


def parse_voc_xml(text: str, *, path: str = "<string>") -> VocAnnotation:
    """Parse one VOC XML document into its verbatim native form.

    Rejects a DOCTYPE outright rather than trusting the parser with it: entity
    declarations live only in a DTD, and `xml.etree` expands internal entities.
    """
    if len(text.encode("utf-8")) > MAX_XML_BYTES:
        raise ConfigError(
            f"{path}: annotation is larger than {MAX_XML_BYTES} bytes. A VOC annotation "
            "for one image is a few hundred bytes; this is not one."
        )
    if "<!DOCTYPE" in text:
        raise ConfigError(
            f"{path}: contains a DOCTYPE declaration. VOC annotations have no legitimate "
            "use for one, and a DTD is the only place an entity can be declared - which "
            "is how an XML document exhausts memory during parsing. Refusing to parse it."
        )

    try:
        root = ElementTree.fromstring(text)  # noqa: S314 - DOCTYPE refused above
    except ElementTree.ParseError as exc:
        raise ConfigError(f"{path}: is not well-formed XML: {exc}") from exc

    if root.tag != "annotation":
        raise ConfigError(f"{path}: root element is <{root.tag}>, expected <annotation>")

    size = root.find("size")
    if size is None:
        raise ConfigError(f"{path}: <size> is missing; image dimensions are required")

    objects: list[VocObject] = []
    for index, node in enumerate(root.findall("object")):
        box = node.find("bndbox")
        if box is None:
            raise ConfigError(f"{path}: object {index} has no <bndbox>")
        objects.append(
            VocObject(
                name=_text(node, "name", path=path),
                xmin=_int(box, "xmin", path=path),
                ymin=_int(box, "ymin", path=path),
                xmax=_int(box, "xmax", path=path),
                ymax=_int(box, "ymax", path=path),
                truncated=_optional_int(node, "truncated", 0),
                difficult=_optional_int(node, "difficult", 0),
                pose=(node.findtext("pose") or "").strip() or None,
            )
        )

    folder = (root.findtext("folder") or "").strip() or None
    return VocAnnotation(
        filename=_text(root, "filename", path=path),
        width=_int(size, "width", path=path),
        height=_int(size, "height", path=path),
        objects=tuple(objects),
        folder=folder,
        depth=_optional_int(size, "depth", 3),
    )


def write_voc_xml(annotation: VocAnnotation) -> str:
    """Serialise back to VOC XML. Exists so the converter can be round-trip tested."""
    lines = ["<annotation>"]
    if annotation.folder is not None:
        lines.append(f"  <folder>{annotation.folder}</folder>")
    lines += [
        f"  <filename>{annotation.filename}</filename>",
        "  <size>",
        f"    <width>{annotation.width}</width>",
        f"    <height>{annotation.height}</height>",
        f"    <depth>{annotation.depth}</depth>",
        "  </size>",
    ]
    for obj in annotation.objects:
        lines.append("  <object>")
        lines.append(f"    <name>{obj.name}</name>")
        if obj.pose is not None:
            lines.append(f"    <pose>{obj.pose}</pose>")
        lines += [
            f"    <truncated>{obj.truncated}</truncated>",
            f"    <difficult>{obj.difficult}</difficult>",
            "    <bndbox>",
            f"      <xmin>{obj.xmin}</xmin>",
            f"      <ymin>{obj.ymin}</ymin>",
            f"      <xmax>{obj.xmax}</xmax>",
            f"      <ymax>{obj.ymax}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    lines.append("</annotation>")
    return "\n".join(lines) + "\n"


def voc_box_to_xyxy(
    obj: VocObject, *, index_base: VocIndexBase = VocIndexBase.ZERO
) -> tuple[float, float, float, float]:
    """One VOC `<bndbox>` to canonical xyxy, under the chosen index base."""
    if index_base is VocIndexBase.ZERO:
        return float(obj.xmin), float(obj.ymin), float(obj.xmax), float(obj.ymax)
    # 1-based with an inclusive max: shift the origin down and widen by the inclusive
    # pixel, so a box from 1 to 1 is one pixel wide rather than zero.
    return float(obj.xmin - 1), float(obj.ymin - 1), float(obj.xmax), float(obj.ymax)


def xyxy_to_voc_box(
    x1: float, y1: float, x2: float, y2: float, *, index_base: VocIndexBase = VocIndexBase.ZERO
) -> tuple[int, int, int, int]:
    """The exact inverse of `voc_box_to_xyxy`."""
    if index_base is VocIndexBase.ZERO:
        return round(x1), round(y1), round(x2), round(y2)
    return round(x1) + 1, round(y1) + 1, round(x2), round(y2)


def map_class(token: str) -> ObjectClass:
    """VOC `<name>` to a Tayr class. Unrecognised becomes UNKNOWN, never DRONE."""
    return CLASS_MAP.get(token.strip().lower(), ObjectClass.UNKNOWN)


def voc_to_video(
    annotation: VocAnnotation,
    *,
    source_video: str,
    index_base: VocIndexBase = VocIndexBase.ZERO,
    image_prefix: str = "",
) -> VideoAnnotation:
    """One VOC file to one canonical `VideoAnnotation` holding a single frame.

    A detection subset is a bag of stills. Modelling each image as its own one-frame
    "video" is the honest encoding: it carries no ordering, no adjacency, and therefore
    no track. The alternative - one video holding all 5,200 images - would assert a
    temporal sequence that does not exist and would make every image a single group for
    split purposes.

    `track_id_source` is `NONE`, so `take_census` counts zero tracks and says why.
    """
    boxes: list[BoxAnnotation] = []
    for obj in annotation.objects:
        x1, y1, x2, y2 = voc_box_to_xyxy(obj, index_base=index_base)
        if x2 <= x1 or y2 <= y1:
            raise ConfigError(
                f"{annotation.filename}: object {obj.name!r} has a non-positive extent "
                f"after conversion: xmin={obj.xmin} xmax={obj.xmax} ymin={obj.ymin} "
                f"ymax={obj.ymax} under index_base={index_base.value}. A degenerate box "
                "is a converter or annotation bug, and clamping it would hide both."
            )
        boxes.append(BoxAnnotation(x1=x1, y1=y1, x2=x2, y2=y2, label=map_class(obj.name)))

    return VideoAnnotation(
        source_video=source_video,
        frames=(
            FrameAnnotation(
                frame_index=0,
                objects=tuple(boxes),
                file_name=f"{image_prefix}{annotation.filename}",
            ),
        ),
        track_id_source=TrackIdSource.NONE,
        width=annotation.width,
        height=annotation.height,
    )


def video_to_voc(
    video: VideoAnnotation,
    *,
    index_base: VocIndexBase = VocIndexBase.ZERO,
    class_names: dict[ObjectClass, str] | None = None,
    folder: str | None = None,
) -> VocAnnotation:
    """Canonical back to native, for the round-trip test.

    Four VOC fields do not survive the trip and that is deliberate: `truncated`, `pose`,
    `difficult`, and `<depth>`. The canonical schema exists to hold what the research
    question needs - where objects are and what they are - and carrying VOC's evaluation
    flags through every downstream stage that ignores them would be dead weight. What is
    NOT lost is any of the geometry, so `difficult` objects are converted as ordinary
    positives; `census` reports how many there were so that choice is visible.
    """
    if len(video.frames) != 1:
        raise ConfigError(
            f"{video.source_video}: a VOC annotation describes one image, but this "
            f"video holds {len(video.frames)} frames."
        )
    if video.width is None or video.height is None:
        raise ConfigError(f"{video.source_video}: image dimensions are required for VOC output")

    names = class_names or {}
    frame = video.frames[0]
    objects: list[VocObject] = []
    for obj in frame.objects:
        xmin, ymin, xmax, ymax = xyxy_to_voc_box(
            obj.x1, obj.y1, obj.x2, obj.y2, index_base=index_base
        )
        objects.append(
            VocObject(
                name=names.get(obj.label, obj.label.value),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
    return VocAnnotation(
        filename=(frame.file_name or f"{video.source_video}.jpg").rsplit("/", 1)[-1],
        width=video.width,
        height=video.height,
        objects=tuple(objects),
        folder=folder,
    )


def gather_index_base_evidence(annotations: list[VocAnnotation]) -> IndexBaseEvidence:
    """Measure what the files say about their own index base."""
    minimums: list[int] = []
    zeros = 0
    at_edge = 0
    n_boxes = 0
    for annotation in annotations:
        for obj in annotation.objects:
            n_boxes += 1
            minimums += [obj.xmin, obj.ymin]
            if obj.xmin == 0 or obj.ymin == 0:
                zeros += 1
            if obj.xmax == annotation.width or obj.ymax == annotation.height:
                at_edge += 1
    return IndexBaseEvidence(
        n_boxes=n_boxes,
        min_coordinate=min(minimums) if minimums else None,
        n_zero_minimums=zeros,
        n_max_at_edge=at_edge,
    )
