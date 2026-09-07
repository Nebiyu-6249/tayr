"""Format converters. Each turns one dataset's native annotations into canonical form.

Every converter in this package documents its source format in its module docstring,
with the field order quoted from a primary source and each fact marked VERIFIED or
UNKNOWN. A converter whose format could not be established from a primary source
raises rather than guessing - see `dut_anti_uav`.
"""

from tayr.datasets.converters.anti_uav import parse_anti_uav_json, write_anti_uav_json
from tayr.datasets.converters.drone_vs_bird import (
    parse_drone_vs_bird,
    write_drone_vs_bird,
)
from tayr.datasets.converters.voc import (
    VocAnnotation,
    VocIndexBase,
    VocObject,
    parse_voc_xml,
    voc_to_video,
    write_voc_xml,
)

__all__ = [
    "VocAnnotation",
    "VocIndexBase",
    "VocObject",
    "parse_anti_uav_json",
    "parse_drone_vs_bird",
    "parse_voc_xml",
    "voc_to_video",
    "write_anti_uav_json",
    "write_drone_vs_bird",
    "write_voc_xml",
]
