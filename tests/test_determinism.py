"""Seeding tests."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tayr.determinism import seed_everything


class TestSeeding:
    def test_same_seed_gives_same_numpy_draws(self) -> None:
        seed_everything(123)
        a = np.random.rand(8)
        seed_everything(123)
        assert np.array_equal(a, np.random.rand(8))

    def test_same_seed_gives_same_python_draws(self) -> None:
        # S311 is intentionally suppressed on these two lines only: this test exists to
        # prove `random` is seeded reproducibly. S311 stays enabled everywhere else,
        # because anything security-bearing (tokens, session ids, reset links) must come
        # from `secrets`, never from `random`.
        seed_everything(123)
        a = [random.random() for _ in range(8)]  # noqa: S311
        seed_everything(123)
        assert a == [random.random() for _ in range(8)]  # noqa: S311

    def test_different_seeds_differ(self) -> None:
        seed_everything(1)
        a = np.random.rand(8)
        seed_everything(2)
        assert not np.array_equal(a, np.random.rand(8))

    def test_report_is_honest_about_torch(self) -> None:
        """The report must state what actually happened, not what we hoped."""
        r = seed_everything(5)
        assert r.seed == 5
        assert r.numpy_seeded and r.python_seeded
        try:
            import torch  # noqa: F401
        except ImportError:
            assert r.torch_seeded is False
            assert "not installed" in r.note
        else:
            assert r.torch_seeded is True

    @pytest.mark.parametrize("bad", [-1, 2**32, 2**40])
    def test_rejects_out_of_range_seed(self, bad: int) -> None:
        with pytest.raises(ValueError, match="seed must be"):
            seed_everything(bad)


class TestStrictTorch:
    def test_strict_mode_raises_when_torch_missing(self) -> None:
        """Training entry points must not silently proceed without torch."""
        try:
            import torch  # noqa: F401
        except ImportError:
            from tayr.errors import DependencyUnavailableError

            with pytest.raises(DependencyUnavailableError, match="torch is required"):
                seed_everything(1, strict_torch=True)
        else:
            pytest.skip("torch is installed; strict path not exercised")
