"""Build the split tree a detector trains from.

RF-DETR's `roboflow` loader wants a root holding `train/`, `valid/` and `test/`, each
with an `_annotations.coco.json` and the images the file names refer to
`[VERIFIED: rfdetr 1.9.4, rfdetr/datasets/coco.py:1265-1268 - note `val` maps onto the
directory named `valid`]`. Datasets do not arrive in that shape; DUT Anti-UAV's
detection subset is `<split>/xml/` beside `<split>/img/`, and its middle split is
called `val`.

**Images are linked, not copied.** Torchvision's `CocoDetection` opens
`os.path.join(root, file_name)` `[VERIFIED: torchvision/datasets/coco.py _load_image,
and rfdetr subclasses it without overriding that method]`, so a `file_name` of
`img/02425.jpg` under a root of `<dest>/train` resolves through a symlink at
`<dest>/train/img`. That means preparing a 10,000-image dataset writes three JSON files
and three links, in about a second, and there is exactly one copy of the licensed
imagery on disk. `--copy` is available for a filesystem that cannot do links.

**The output is refused inside the repository.** Converted annotations are text, so the
`licence-guard` CI job - which looks for imagery and weights - would not catch them. A
dataset under a no-redistribution agreement whose boxes get committed is a licence
breach that no automated check downstream would notice, so the check is here.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tayr.datasets.coco import to_coco
from tayr.datasets.converters.voc import DEFAULT_INDEX_BASE, VocIndexBase
from tayr.datasets.loader import VOC_IMG_DIR, load_native_directory
from tayr.datasets.schema import DatasetAnnotation
from tayr.errors import ConfigError

#: Source split name -> the directory name RF-DETR reads it from.
SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

#: What RF-DETR's roboflow loader globs for inside each split directory.
ANNOTATION_FILENAME = "_annotations.coco.json"


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """One split written into the detector tree."""

    source_split: str
    destination: Path
    annotation_path: Path
    n_images: int
    n_boxes: int
    n_empty_images: int
    image_link: Path | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Everything `prepare_detector_dataset` wrote."""

    root: Path
    splits: tuple[PreparedSplit, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        lines = [f"Detector dataset written to {self.root}", "=" * 60]
        for split in self.splits:
            lines.append(
                f"  {split.source_split:>6s} -> {split.destination.name:6s}  "
                f"{split.n_images:6d} image(s)  {split.n_boxes:6d} box(es)  "
                f"{split.n_empty_images:5d} with none"
            )
        lines += [
            "",
            "  Point train.dataset_dir at the root above.",
        ]
        if self.warnings:
            lines += ["", "  WARNINGS:"] + [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def _repository_root(start: Path) -> Path | None:
    """The git work tree containing `start`, or None."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return Path(out.stdout.strip())


def refuse_output_inside_a_repository(destination: Path, dataset: DatasetAnnotation) -> None:
    """Raise if non-redistributable annotations would land in a git work tree."""
    if dataset.redistributable:
        return
    probe = destination if destination.exists() else destination.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    repo = _repository_root(probe)
    if repo is None:
        return
    raise ConfigError(
        f"refusing to write derived annotations for {dataset.name!r} into {destination}, "
        f"which is inside the git work tree at {repo}. That dataset is marked "
        "non-redistributable, converted annotations are text so the licence-guard CI job "
        "would not catch them, and a commit of them is a licence breach. Write to a path "
        "outside any repository - the `data/` convention in configs/baseline.yaml is "
        "gitignored but this check does not rely on that."
    )


def _link_images(source: Path, destination: Path, *, copy: bool) -> Path | None:
    """Make the split's images reachable from `destination`. Returns the link path."""
    if not source.is_dir():
        raise ConfigError(f"no image directory at {source}")
    target = destination / VOC_IMG_DIR
    if target.exists() or target.is_symlink():
        raise ConfigError(
            f"{target} already exists. Refusing to replace it: if a previous prepare "
            "wrote there, delete the destination and run again rather than half-updating it."
        )
    if copy:
        import shutil

        shutil.copytree(source, target)
        return target
    target.symlink_to(os.path.relpath(source, destination), target_is_directory=True)
    return target


def prepare_split(
    source: Path,
    *,
    split: str,
    destination_root: Path,
    fmt: str,
    name: str,
    copy_images: bool = False,
    index_base: VocIndexBase = DEFAULT_INDEX_BASE,
) -> PreparedSplit:
    """Convert one split and write it into the detector tree."""
    if split not in SPLIT_DIRS:
        raise ConfigError(
            f"unknown split {split!r}; expected one of {', '.join(sorted(SPLIT_DIRS))}"
        )
    dataset = load_native_directory(source, fmt=fmt, name=name, split=split, index_base=index_base)
    refuse_output_inside_a_repository(destination_root, dataset)

    missing = [note for note in dataset.notes if "not in" in note and "annotation(s) name" in note]
    if missing:
        raise ConfigError(
            f"{source}: {missing[0]} Refusing to write a training set whose annotations "
            "point at images that are not there - the failure would otherwise surface "
            "as a crash partway through the first epoch."
        )

    destination = destination_root / SPLIT_DIRS[split]
    destination.mkdir(parents=True, exist_ok=True)
    link = _link_images(source / VOC_IMG_DIR, destination, copy=copy_images)

    annotation_path = destination / ANNOTATION_FILENAME
    annotation_path.write_text(json.dumps(to_coco(dataset), indent=1), encoding="utf-8")

    return PreparedSplit(
        source_split=split,
        destination=destination,
        annotation_path=annotation_path,
        n_images=dataset.n_frames,
        n_boxes=dataset.n_boxes,
        n_empty_images=sum(v.n_empty_frames for v in dataset.videos),
        image_link=link,
        notes=dataset.notes,
    )


def prepare_detector_dataset(
    source_root: Path,
    *,
    destination_root: Path,
    fmt: str = "voc",
    name: str = "unnamed",
    splits: tuple[str, ...] = ("train", "val", "test"),
    copy_images: bool = False,
    index_base: VocIndexBase = DEFAULT_INDEX_BASE,
) -> PreparedDataset:
    """Build `train/`, `valid/` and `test/` from a dataset's own split directories.

    A missing source split is a warning rather than an error: a dataset with no test
    split is a real situation, and RF-DETR falls back to `valid` for evaluation. A
    missing *train* split is fatal, because there is then nothing to prepare.
    """
    if not source_root.is_dir():
        raise ConfigError(f"not a directory: {source_root}")

    prepared: list[PreparedSplit] = []
    warnings: list[str] = []
    for split in splits:
        source = source_root / split
        if not source.is_dir():
            if split == "train":
                raise ConfigError(
                    f"no train split at {source}. Expected the dataset root to hold "
                    f"{', '.join(splits)} subdirectories."
                )
            warnings.append(f"no {split}/ split under {source_root}; skipped.")
            continue
        prepared.append(
            prepare_split(
                source,
                split=split,
                destination_root=destination_root,
                fmt=fmt,
                name=name,
                copy_images=copy_images,
                index_base=index_base,
            )
        )

    # Loader findings are per split but usually identical; report each distinct one once.
    seen: set[str] = set()
    for written in prepared:
        for note in written.notes:
            if note not in seen:
                seen.add(note)
                warnings.append(f"[{written.source_split}] {note}")

    return PreparedDataset(root=destination_root, splits=tuple(prepared), warnings=tuple(warnings))
