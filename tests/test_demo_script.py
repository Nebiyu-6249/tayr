"""The demo script claims a duration. This checks the claim against its own text.

A two-minute script whose spoken text takes four minutes to say is not a two-minute
script, and that failure is invisible on inspection — you only find it by counting. The
first draft of `docs/DEMO_SCRIPT.md` was 334 words per minute. These tests exist so that
cannot come back silently after an edit.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "DEMO_SCRIPT.md"

#: Words per minute the script is budgeted at. A measured, unrushed narration pace;
#: broadcast news sits around 150-180. [ASSUMED] - not fitted to a recording.
BUDGET_WPM = 175.0

#: Total runtime the document claims in its title and timing table.
CLAIMED_SECONDS = 120

# Escapes rather than literals: the separators are U+00B7, U+2014 and U+2013, which are
# indistinguishable from ASCII in most editors and would make this pattern silently
# unmatchable if one were retyped.
_HEADER = re.compile(r"^## (\d+) \u00b7 (.+?) \u2014 (\d):(\d\d)\u2013(\d):(\d\d)$", re.MULTILINE)
_OPTIONAL = "**Add back only if"


def _seconds(minutes: str, secs: str) -> int:
    return int(minutes) * 60 + int(secs)


class Segment:
    __slots__ = ("end", "index", "start", "title", "words")

    def __init__(self, index: int, title: str, start: int, end: int, body: str) -> None:
        self.index = index
        self.title = title
        self.start = start
        self.end = end
        # Lines starting "> " are spoken. Everything after the optional-lines marker is
        # explicitly outside the budget, so it does not count.
        spoken = body.split(_OPTIONAL)[0]
        text = " ".join(line[2:] for line in spoken.splitlines() if line.startswith("> "))
        self.words = len(re.sub(r"[*_`\[\]]", "", text).split())

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def wpm(self) -> float:
        return self.words / self.duration * 60


def _segments() -> list[Segment]:
    text = SCRIPT.read_text(encoding="utf-8")
    matches = list(_HEADER.finditer(text))
    assert matches, "no timed segments found - did the header format change?"
    out: list[Segment] = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(
            Segment(
                index=int(m.group(1)),
                title=m.group(2),
                start=_seconds(m.group(3), m.group(4)),
                end=_seconds(m.group(5), m.group(6)),
                body=text[m.end() : body_end],
            )
        )
    return out


def test_segments_are_contiguous_and_fill_the_claimed_runtime() -> None:
    segments = _segments()
    assert [s.index for s in segments] == list(range(1, len(segments) + 1))
    assert segments[0].start == 0
    for earlier, later in pairwise(segments):
        assert later.start == earlier.end, f"gap or overlap before segment {later.index}"
    assert segments[-1].end == CLAIMED_SECONDS


def test_total_spoken_text_fits_the_claimed_runtime() -> None:
    segments = _segments()
    words = sum(s.words for s in segments)
    allowed = BUDGET_WPM * CLAIMED_SECONDS / 60
    assert words <= allowed, (
        f"{words} spoken words needs {words / BUDGET_WPM * 60:.0f}s at {BUDGET_WPM:.0f} wpm, "
        f"but the script claims {CLAIMED_SECONDS}s (budget {allowed:.0f} words)"
    )


@pytest.mark.parametrize("segment", _segments(), ids=lambda s: f"{s.index}-{s.title}")
def test_each_segment_fits_its_own_slot(segment: Segment) -> None:
    assert segment.wpm <= BUDGET_WPM, (
        f"segment {segment.index} ({segment.title}): {segment.words} words in "
        f"{segment.duration}s is {segment.wpm:.0f} wpm"
    )


@pytest.mark.parametrize("segment", _segments(), ids=lambda s: f"{s.index}-{s.title}")
def test_timing_table_word_counts_match_the_script(segment: Segment) -> None:
    """The table is a promise about the text below it. Keep them in step."""
    text = SCRIPT.read_text(encoding="utf-8")
    row = re.search(rf"^\| {segment.index} \| .+? \| .+? \| .+? \| (\d+) \|$", text, re.MULTILINE)
    assert row is not None, f"segment {segment.index} has no row in the timing table"
    assert int(row.group(1)) == segment.words, (
        f"timing table says {row.group(1)} words for segment {segment.index}, "
        f"text has {segment.words}"
    )


#: Segment 2 must speak one of these, and which one depends on the command being
#: recorded. `tayr watch demo` scripts its detections; `tayr watch run` detects for real
#: but still has no fitted classifier. Both disclose provenance and both are true of the
#: run they belong to - speaking neither is the failure this guards against.
PROVENANCE_DISCLOSURES = (
    "scripted, not detected",
    "the classifier is not",
)


def test_a_provenance_disclosure_is_spoken_not_just_documented() -> None:
    """CLAUDE.md 1.3: provenance is labelled wherever it appears. On camera counts."""
    spoken = next(s for s in _segments() if s.index == 2)
    text = SCRIPT.read_text(encoding="utf-8")
    body = text.split("## 2 · ")[1].split("\n## ")[0]
    said = " ".join(line[2:] for line in body.splitlines() if line.startswith("> "))
    assert any(d in said for d in PROVENANCE_DISCLOSURES), (
        "segment 2 must speak a provenance disclosure - either that the detections are "
        f"scripted, or that the classifier is untrained. Says neither: {said!r}"
    )
    assert spoken.words > 0
