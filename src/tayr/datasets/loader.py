"""Load a directory of native annotation files into canonical form.

Directory conventions per format:

  dvb      one `*.txt` per video, filename stem is the video id.
  antiuav  one subdirectory per sequence, each containing `IR_label.json`;
           the subdirectory name is the video id. This matches the layout the
           benchmark's own toolkit globs for
           `[VERIFIED: antiuav410.py line 38, glob '*/IR_label.json']`.
"""

from __future__ import annotations

from pathlib import Path

from tayr.datasets.converters.anti_uav import parse_anti_uav_json
from tayr.datasets.converters.drone_vs_bird import parse_drone_vs_bird
from tayr.datasets.schema import DatasetAnnotation, VideoAnnotation
from tayr.errors import ConfigError

SUPPORTED_FORMATS = ("dvb", "antiuav")

# Licence and redistribution rights per format, from Phase 0 verification. These
# defaults are conservative: a dataset is assumed non-redistributable unless its
# licence was confirmed to permit it.
_LICENCE: dict[str, tuple[str, bool]] = {
    # Data usage agreement, research use only, no redistribution rights granted.
    "dvb": ("WOSDETC data usage agreement (no redistribution)", False),
    # MIT, but we still never place dataset-derived files in the public repo.
    "antiuav": ("MIT", False),
}


def load_native_directory(directory: Path, *, fmt: str, name: str, split: str) -> DatasetAnnotation:
    """Parse every annotation file under `directory` in the named native format."""
    if fmt not in SUPPORTED_FORMATS:
        raise ConfigError(
            f"unsupported format {fmt!r}; expected one of {', '.join(SUPPORTED_FORMATS)}. "
            "DUT Anti-UAV is deliberately unimplemented - its format has not been "
            "established from a primary source (see converters/dut_anti_uav.py)."
        )
    if not directory.is_dir():
        raise ConfigError(f"not a directory: {directory}")

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
