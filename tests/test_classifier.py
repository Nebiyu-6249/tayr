"""Classifier and hypothesis-evaluation tests.

The evaluation tests matter more than the model tests: they encode what this project is
allowed to claim from a given amount of data.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.classify.evaluation import (
    ArmResult,
    BucketScore,
    compare_arms,
    wilson_interval,
)
from tayr.classify.motion import MIN_TRACKS_TO_TRAIN, MotionClassifier, NotEnoughDataError
from tayr.errors import TayrError
from tayr.geometry import SizeBucket
from tayr.tracking.features import extract_features

pytest.importorskip("sklearn", reason="scikit-learn is in the 'cv' extra")


def synth_track(kind: str, seed: int, n: int = 60, fps: float = 30.0):  # type: ignore[no-untyped-def]
    """A SYNTHETIC trajectory. Constructed to differ; proves nothing about real classes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    if kind == "flapper":
        x = t * 3.0 + rng.normal(0, 0.2, n)
        y = 300.0 + 8.0 * np.sin(2 * np.pi * 5 * t / fps) + rng.normal(0, 0.2, n)
    else:  # "hoverer"
        x = np.full(n, 500.0) + rng.normal(0, 0.1, n)
        y = np.full(n, 300.0) + rng.normal(0, 0.1, n)
    side = 20.0
    boxes = np.stack([x - side / 2, y - side / 2, x + side / 2, y + side / 2], axis=-1)
    return extract_features(boxes, np.arange(n, dtype=np.int64), fps=fps)


class TestWilsonInterval:
    def test_zero_trials_gives_full_uncertainty(self) -> None:
        """With no data the true value could be anything; a narrow interval would lie."""
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_interval_narrows_with_more_data(self) -> None:
        small = wilson_interval(8, 10)
        large = wilson_interval(800, 1000)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_stays_inside_zero_one_at_the_extremes(self) -> None:
        """The normal approximation produces bounds outside [0,1] here."""
        for successes, trials in ((0, 5), (5, 5), (0, 1), (1, 1)):
            low, high = wilson_interval(successes, trials)
            assert 0.0 <= low <= high <= 1.0

    def test_perfect_score_on_few_trials_is_not_certainty(self) -> None:
        """20/20 correct does not mean 100%. This is the arithmetic that stops a small
        clean result being quoted as a settled one."""
        low, _ = wilson_interval(20, 20)
        assert low < 0.90

    def test_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError, match="within"):
            wilson_interval(11, 10)


class TestHypothesisVerdict:
    @staticmethod
    def _arms(a_correct: int, m_correct: int, n: int = 200):  # type: ignore[no-untyped-def]
        appearance = ArmResult(
            "appearance", {SizeBucket.TINY: BucketScore(SizeBucket.TINY, a_correct, n)}
        )
        motion = ArmResult("motion", {SizeBucket.TINY: BucketScore(SizeBucket.TINY, m_correct, n)})
        return appearance, motion

    def test_clear_motion_win(self) -> None:
        result = compare_arms(*self._arms(a_correct=100, m_correct=180))
        assert result.motion_wins_in_small_buckets() is True

    def test_clear_appearance_win_falsifies_the_hypothesis(self) -> None:
        """A negative result is a real result and must be reportable as one."""
        result = compare_arms(*self._arms(a_correct=180, m_correct=100))
        assert result.motion_wins_in_small_buckets() is False
        assert "HYPOTHESIS IS WRONG" in result.render()

    def test_overlapping_intervals_are_undetermined_not_negative(self) -> None:
        """'We could not tell' is a distinct answer from 'no'. Collapsing them would
        turn an under-powered experiment into a negative result."""
        result = compare_arms(*self._arms(a_correct=10, m_correct=12, n=20))
        assert result.motion_wins_in_small_buckets() is None
        assert "UNDETERMINED" in result.render()
        assert "under-powered" in result.render()

    def test_empty_buckets_are_undetermined(self) -> None:
        result = compare_arms(ArmResult("appearance"), ArmResult("motion"))
        assert result.motion_wins_in_small_buckets() is None

    def test_thin_buckets_are_flagged_in_the_report(self) -> None:
        result = compare_arms(*self._arms(a_correct=5, m_correct=8, n=10))
        assert "*" in result.render()
        assert "the interval, not the point estimate" in result.render()

    def test_mismatched_track_counts_break_the_control(self) -> None:
        """The comparison is only controlled if both arms classify the SAME tracks."""
        appearance = ArmResult(
            "appearance", {SizeBucket.TINY: BucketScore(SizeBucket.TINY, 40, 50)}
        )
        motion = ArmResult("motion", {SizeBucket.TINY: BucketScore(SizeBucket.TINY, 60, 70)})
        result = compare_arms(appearance, motion)
        assert any("SAME tracks" in n for n in result.notes)

    def test_synthetic_runs_are_labelled(self) -> None:
        result = compare_arms(ArmResult("a"), ArmResult("m"), synthetic=True)
        assert "SYNTHETIC DATA" in result.render()


class TestMotionClassifier:
    def _dataset(self, n_per_class: int = 15):  # type: ignore[no-untyped-def]
        features, labels, groups = [], [], []
        for i in range(n_per_class):
            features.append(synth_track("flapper", seed=i))
            labels.append("bird")
            groups.append(f"video{i}")
            features.append(synth_track("hoverer", seed=1000 + i))
            labels.append("drone")
            groups.append(f"video{i}")
        return features, labels, groups

    def test_trains_and_predicts_on_synthetic_archetypes(self) -> None:
        """SYNTHETIC trajectories built to differ. This shows the plumbing works, not
        that real birds and drones separate."""
        clf = MotionClassifier()
        report = clf.fit(*self._dataset())
        assert clf.is_trained
        assert report.n_classes == 2
        assert set(clf.classes) == {"bird", "drone"}
        preds = clf.predict([synth_track("hoverer", seed=9999)])
        assert preds[0] in {"bird", "drone"}

    def test_reports_feature_importances(self) -> None:
        """Interpretability is why trees were chosen. A model that says 'drone' without
        saying which motion property drove it is little use in a dissertation."""
        report = MotionClassifier().fit(*self._dataset())
        assert report.feature_importances
        assert set(report.feature_importances) <= set(
            __import__(
                "tayr.tracking.features", fromlist=["TrackFeatures"]
            ).TrackFeatures.feature_names()
        )

    def test_refuses_to_train_on_too_few_tracks(self) -> None:
        """Training anyway produces a model whose accuracy is an artefact of the split."""
        features, labels, groups = self._dataset(n_per_class=3)
        with pytest.raises(NotEnoughDataError, match="too few"):
            MotionClassifier().fit(features, labels, groups)

    def test_refuses_a_single_class(self) -> None:
        """One class reports 100% accuracy and has learned nothing."""
        features = [synth_track("flapper", seed=i) for i in range(MIN_TRACKS_TO_TRAIN + 2)]
        labels = ["drone"] * len(features)
        groups = [f"v{i}" for i in range(len(features))]
        with pytest.raises(NotEnoughDataError, match="only one class"):
            MotionClassifier().fit(features, labels, groups)

    def test_warns_when_too_few_groups_for_grouped_cv(self) -> None:
        features, labels, _ = self._dataset()
        groups = ["only-one-video"] * len(features)
        report = MotionClassifier().fit(features, labels, groups)
        assert any("grouped cross-validation" in w for w in report.warnings)

    def test_mismatched_input_lengths(self) -> None:
        features, labels, groups = self._dataset()
        with pytest.raises(ValueError, match="correspond"):
            MotionClassifier().fit(features, labels[:-1], groups)

    def test_predicting_before_training_raises(self) -> None:
        with pytest.raises(TayrError, match="not trained"):
            MotionClassifier().predict([synth_track("flapper", seed=1)])

    def test_training_is_deterministic_given_a_seed(self) -> None:
        a, b = MotionClassifier(seed=7), MotionClassifier(seed=7)
        data = self._dataset()
        a.fit(*data)
        b.fit(*data)
        probe = [synth_track("flapper", seed=555)]
        assert a.predict(probe) == b.predict(probe)
