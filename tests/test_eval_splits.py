"""Grouped-split tests.

Frame-level splitting is the single easiest way to accidentally fabricate a good
result, so these check that splitting is whole-video, deterministic, and stable.
"""

from __future__ import annotations

import pytest

from tayr.errors import ConfigError
from tayr.eval.splits import grouped_split


def videos(n: int) -> list[str]:
    return [f"clip{i:03d}" for i in range(n)]


class TestGrouping:
    def test_every_video_lands_in_exactly_one_split(self) -> None:
        s = grouped_split(videos(100), seed=1)
        all_assigned = list(s.train) + list(s.val) + list(s.test)
        assert sorted(all_assigned) == sorted(videos(100))
        assert len(set(all_assigned)) == 100

    def test_splits_are_disjoint(self) -> None:
        s = grouped_split(videos(50), seed=7)
        assert not (set(s.train) & set(s.val))
        assert not (set(s.train) & set(s.test))
        assert not (set(s.val) & set(s.test))

    def test_no_split_is_empty(self) -> None:
        for n in (3, 4, 5, 10, 37):
            s = grouped_split(videos(n), seed=3)
            assert s.train and s.val and s.test, f"empty split at n={n}"

    def test_approximate_fractions(self) -> None:
        s = grouped_split(videos(1000), seed=11, train_fraction=0.7, val_fraction=0.15)
        assert 690 <= len(s.train) <= 710
        assert 140 <= len(s.val) <= 160
        assert 140 <= len(s.test) <= 160

    def test_split_of_lookup(self) -> None:
        s = grouped_split(videos(20), seed=2)
        assert s.split_of(s.train[0]) == "train"
        assert s.split_of(s.test[0]) == "test"
        with pytest.raises(ConfigError, match="not assigned"):
            s.split_of("never-seen")


class TestDeterminism:
    def test_same_seed_same_assignment(self) -> None:
        assert grouped_split(videos(40), seed=5) == grouped_split(videos(40), seed=5)

    def test_different_seed_different_assignment(self) -> None:
        assert grouped_split(videos(40), seed=5).train != grouped_split(videos(40), seed=6).train

    def test_input_order_does_not_matter(self) -> None:
        """Assignment must depend on the video name, not the order files were listed -
        otherwise a differently-sorted directory read silently changes the split."""
        forward = grouped_split(videos(30), seed=9)
        backward = grouped_split(list(reversed(videos(30))), seed=9)
        assert forward == backward

    def test_adding_a_video_does_not_reshuffle_the_others(self) -> None:
        """A hash-based assignment keeps earlier runs comparable. A shuffle would move
        many videos across splits when one is added, invalidating any comparison."""
        before = grouped_split(videos(200), seed=4)
        after = grouped_split([*videos(200), "clip999"], seed=4)
        moved = sum(1 for v in videos(200) if before.split_of(v) != after.split_of(v))
        assert moved <= 3, f"{moved} videos changed split after adding one"


class TestRejects:
    def test_empty_input(self) -> None:
        with pytest.raises(ConfigError, match="empty list"):
            grouped_split([], seed=1)

    def test_duplicate_video_names(self) -> None:
        """A duplicate could otherwise land in two splits at once."""
        with pytest.raises(ConfigError, match="duplicate source_video"):
            grouped_split(["a", "b", "a", "c"], seed=1)

    def test_too_few_videos_to_hold_anything_out(self) -> None:
        with pytest.raises(ConfigError, match="at least 3 videos"):
            grouped_split(["a", "b"], seed=1)

    @pytest.mark.parametrize(("train", "val"), [(0.0, 0.15), (1.0, 0.0), (0.9, 0.15), (0.7, -0.1)])
    def test_invalid_fractions(self, train: float, val: float) -> None:
        with pytest.raises(ConfigError):
            grouped_split(videos(10), seed=1, train_fraction=train, val_fraction=val)
