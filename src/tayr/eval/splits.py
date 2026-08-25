"""Train/val/test splitting, grouped by source video.

Frames from one video are enormously correlated: consecutive frames of the same drone
against the same sky are near-duplicates. Splitting at the frame level puts
near-duplicates on both sides of the train/test boundary, and every reported number
becomes an overestimate - often a dramatic one.

So splitting happens at the **video** level, always. `EvalConfig` refuses to disable it
and there is no frame-level path in this module to fall back to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from tayr.errors import ConfigError


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Which videos landed in which split."""

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    _lookup: dict[str, str] = field(default_factory=dict, repr=False)

    def split_of(self, source_video: str) -> str:
        try:
            return self._lookup[source_video]
        except KeyError:
            raise ConfigError(f"{source_video!r} was not assigned to any split") from None

    @property
    def n_videos(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def _stable_rank(source_video: str, seed: int) -> int:
    """Deterministic per-video ordering that does not depend on input order.

    A hash rather than a shuffle so that adding one video to a dataset does not
    reshuffle every other video across splits, which would silently invalidate any
    comparison against an earlier run.
    """
    digest = hashlib.sha256(f"{seed}:{source_video}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def grouped_split(
    source_videos: list[str],
    *,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> SplitAssignment:
    """Assign whole videos to train/val/test.

    Deterministic given `seed` and the set of video names, and stable under insertion:
    adding a video reassigns only that video, not the whole dataset.
    """
    if not source_videos:
        raise ConfigError("cannot split an empty list of videos")

    unique = sorted(set(source_videos))
    if len(unique) != len(source_videos):
        raise ConfigError(
            f"duplicate source_video names: {len(source_videos)} given, {len(unique)} unique. "
            "A video appearing twice could otherwise land in two splits at once."
        )

    if not 0.0 < train_fraction < 1.0:
        raise ConfigError(f"train_fraction must be in (0, 1); got {train_fraction}")
    if not 0.0 <= val_fraction < 1.0:
        raise ConfigError(f"val_fraction must be in [0, 1); got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ConfigError(
            f"train_fraction + val_fraction must leave room for a test split; "
            f"got {train_fraction} + {val_fraction} = {train_fraction + val_fraction}"
        )

    if len(unique) < 3:
        raise ConfigError(
            f"need at least 3 videos to form train/val/test, got {len(unique)}. "
            "With fewer, report on the whole set and say so rather than pretending to "
            "have held anything out."
        )

    ordered = sorted(unique, key=lambda v: _stable_rank(v, seed))
    n = len(ordered)
    n_train = max(1, round(n * train_fraction))
    n_val = max(1, round(n * val_fraction))
    # Guarantee a non-empty test split even at small n.
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        if n_train + n_val >= n:
            n_train = n - n_val - 1

    train = tuple(ordered[:n_train])
    val = tuple(ordered[n_train : n_train + n_val])
    test = tuple(ordered[n_train + n_val :])

    lookup = dict.fromkeys(train, "train") | dict.fromkeys(val, "val") | dict.fromkeys(test, "test")
    return SplitAssignment(train=train, val=val, test=test, _lookup=lookup)
