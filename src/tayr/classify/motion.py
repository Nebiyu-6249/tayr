"""Motion arm: gradient-boosted trees on hand-designed track features.

Trees first, per the brief, and for reasons that hold especially here. They train in
seconds, they are interpretable (feature importances explain *why* a track was called a
drone), and with low hundreds of tracks a deep sequence model is very unlikely to be
trainable at all. If a 1D CNN or GRU is tried later it must beat this baseline on the
same splits, and if it does not, the tree is the result and the writeup says so.

**Splits are grouped by source video, always.** Frames of one track are near-duplicates;
so, often, are separate tracks from the same clip - same sky, same camera, same bird.
Ungrouped cross-validation therefore reports a number that is fiction. `GroupKFold` is
used rather than `KFold` and there is no ungrouped path in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tayr.errors import DependencyUnavailableError, TayrError
from tayr.tracking.features import TrackFeatures

# Below this, cross-validation folds contain too few tracks to mean anything.
MIN_TRACKS_TO_TRAIN = 20


class NotEnoughDataError(TayrError):
    """Too few tracks, or too few groups, to train or evaluate honestly."""


@dataclass(slots=True)
class TrainingReport:
    """What training actually did, recorded for the run manifest."""

    n_tracks: int
    n_groups: int
    n_classes: int
    class_counts: dict[str, int] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class MotionClassifier:
    """Gradient-boosted trees over `TrackFeatures` vectors.

    Uses scikit-learn's `HistGradientBoostingClassifier` rather than xgboost: it is
    already a core dependency via scipy/scikit-learn, needs no extra pin, and at a few
    hundred samples the difference between the two is noise.
    """

    def __init__(self, *, seed: int = 1337, max_iter: int = 200) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self._model: object | None = None
        self._classes: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def fit(
        self,
        features: list[TrackFeatures],
        labels: list[str],
        groups: list[str],
    ) -> TrainingReport:
        """Train on track-level features.

        `groups` is the source video per track. It is required, not optional: without it
        a caller cannot construct a grouped split, and an ungrouped one inflates every
        number that follows.
        """
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise DependencyUnavailableError(
                "scikit-learn is required to train the motion classifier. "
                "Install the 'cv' extra: pip install -e '.[cv]'"
            ) from exc

        if not (len(features) == len(labels) == len(groups)):
            raise ValueError(
                f"{len(features)} feature vector(s), {len(labels)} label(s), "
                f"{len(groups)} group(s) - all three must correspond"
            )
        if len(features) < MIN_TRACKS_TO_TRAIN:
            raise NotEnoughDataError(
                f"{len(features)} track(s) is too few to train on (minimum "
                f"{MIN_TRACKS_TO_TRAIN}). Training anyway would produce a model whose "
                "reported accuracy is an artefact of the split."
            )

        classes = sorted(set(labels))
        if len(classes) < 2:
            raise NotEnoughDataError(
                f"only one class present ({classes[0]!r}). A classifier trained on one "
                "class reports 100% accuracy and has learned nothing."
            )

        X = np.stack([f.as_vector() for f in features])
        y = np.array(labels)

        model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, random_state=self.seed, early_stopping=False
        )
        model.fit(X, y)
        self._model = model
        self._classes = classes

        counts = {c: int(np.sum(y == c)) for c in classes}
        warnings: list[str] = []
        smallest = min(counts.values())
        if smallest < MIN_TRACKS_TO_TRAIN:
            warnings.append(
                f"the smallest class has {smallest} track(s); per-class results from "
                "this few need an interval, and a point estimate alone is an overclaim"
            )
        n_groups = len(set(groups))
        if n_groups < 3:
            warnings.append(
                f"only {n_groups} distinct source video(s); grouped cross-validation "
                "needs at least 3 to form meaningful folds"
            )

        return TrainingReport(
            n_tracks=len(features),
            n_groups=n_groups,
            n_classes=len(classes),
            class_counts=counts,
            feature_importances=self._importances(X, y, model),
            warnings=warnings,
        )

    def predict(self, features: list[TrackFeatures]) -> list[str]:
        if self._model is None:
            raise TayrError("classifier is not trained")
        X = np.stack([f.as_vector() for f in features])
        return [str(label) for label in self._model.predict(X)]  # type: ignore[attr-defined]

    def predict_proba(self, features: list[TrackFeatures]) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise TayrError("classifier is not trained")
        X = np.stack([f.as_vector() for f in features])
        result: npt.NDArray[np.float64] = self._model.predict_proba(X)  # type: ignore[attr-defined]
        return result

    def _importances(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.str_], model: object
    ) -> dict[str, float]:
        """Permutation importance: which features actually carry the signal.

        Interpretability is a reason this baseline was chosen. A model that says 'drone'
        without saying which motion property drove it is not much use in a dissertation.
        """
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            return {}
        try:
            result = permutation_importance(
                model, X, y, n_repeats=5, random_state=self.seed, scoring="accuracy"
            )
        except Exception:  # noqa: BLE001 - importance is diagnostic, never load-bearing
            return {}
        names = TrackFeatures.feature_names()
        return {
            name: float(value) for name, value in zip(names, result.importances_mean, strict=True)
        }
