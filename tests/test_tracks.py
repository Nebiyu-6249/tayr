"""The labelled track dataset, and the one thing it must never be mistaken for.

A row's label says which clip it came from and what a person said that clip contains. It
does not say anyone looked at the track. Every test that touches labelling is really
testing that distinction, because the moment a clip-level assertion is read as ground
truth, every number computed downstream inherits a confidence nobody earned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tayr.errors import ConfigError
from tayr.tracks import (
    MIN_TRACK_SECONDS,
    LabelBasis,
    LabelledTrack,
    TrackStore,
    fit_motion_arm,
    hypothesis_status,
    ingest_run,
    separate,
)
from tayr.tracks.analysis import chance_separation_probability

FEATURES = {
    "mean_speed_px_per_frame": 1.5,
    "speed_variance": 0.2,
    "acceleration_variance": 0.05,
    "vertical_oscillation_hz": 4.2,
    "vertical_oscillation_power": 0.55,
    "heading_entropy": 1.1,
    "hover_fraction": 0.02,
    "trajectory_smoothness": 0.8,
    "scale_change_rate": 0.001,
}


def decision_record(
    track_number: int,
    *,
    duration: float = 5.0,
    features: dict[str, float] | None = None,
    features_available: bool = True,
    pot: float = 14.0,
    job: str = "run-1",
    synthetic: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "track_id": f"{job}-t{track_number}",
        "track_number": track_number,
        "n_observations": 120,
        "duration_seconds": duration,
        "median_pixels_on_target": pot,
        "features_available": features_available,
    }
    if features_available:
        result["features"] = dict(features or FEATURES)
    return {
        "track_id": f"{job}-t{track_number}",
        "job_id": job,
        "site_id": "demo-north",
        "verdict": "escalate",
        "attention": "prompt",
        "uncertainty": "no_classifier_trained",
        "rule_id": "uncertain.no_classifier",
        "synthetic": synthetic,
        "tool_calls": [
            {"tool_name": "analyze_track", "ok": True, "arguments": {}, "result": result}
        ],
    }


def run_dir(tmp_path: Path, records: list[dict[str, Any]], *, name: str = "run") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decisions.json").write_text(json.dumps(records), encoding="utf-8")
    return directory


class TestTheLabelIsProvenanceNotGroundTruth:
    """The property everything downstream rests on."""

    def test_ingestion_can_only_ever_produce_a_clip_level_basis(self, tmp_path: Path) -> None:
        """A bird clip can have an aircraft in shot. Ingestion has a clip-level assertion
        and nothing else, so it must not be able to record anything stronger."""
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1)]), label="bird", asserted_by="nebiyu"
        )
        assert [t.label_basis for t in report.tracks] == [LabelBasis.CLIP_PROVENANCE]

    def test_the_source_never_claims_ingestion_can_confirm_a_track(self) -> None:
        """Read the module: a future edit that upgrades the basis during ingest fails."""
        source = Path("src/tayr/tracks/ingest.py").read_text(encoding="utf-8")
        assert "LabelBasis.CLIP_PROVENANCE" in source
        assert "LabelBasis.OPERATOR_CONFIRMED" not in source

    def test_the_asserter_is_recorded_so_a_disputed_row_has_an_author(self, tmp_path: Path) -> None:
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1)]), label="bird", asserted_by="nebiyu"
        )
        assert report.tracks[0].asserted_by == "nebiyu"

    def test_a_synthetic_run_is_flagged_on_every_row(self, tmp_path: Path) -> None:
        """Placeholder detections must never enter a dataset behind a reported result."""
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1, synthetic=True)]),
            label="drone",
            asserted_by="op",
        )
        assert any("synthetic" in note for note in report.tracks[0].notes)
        assert any("SYNTHETIC RUN" in note for note in report.notes)


class TestIngestRefusesToInvent:
    def test_a_track_without_features_is_excluded_and_counted(self, tmp_path: Path) -> None:
        """Zero-filling would put fabricated rows into the only dataset this has."""
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1), decision_record(2, features_available=False)]),
            label="bird",
            asserted_by="op",
        )
        assert report.n_ingested == 1
        assert report.n_without_features == 1
        assert "zero-filled" in report.render()

    def test_a_short_track_is_excluded_at_the_rules_own_threshold(self, tmp_path: Path) -> None:
        """A dataset of tracks the verdict rules would refuse to judge measures something
        the system never does."""
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1, duration=MIN_TRACK_SECONDS - 0.1)]),
            label="bird",
            asserted_by="op",
        )
        assert report.n_ingested == 0
        assert report.n_too_short == 1

    def test_a_track_exactly_at_the_threshold_is_kept(self, tmp_path: Path) -> None:
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1, duration=MIN_TRACK_SECONDS)]),
            label="bird",
            asserted_by="op",
        )
        assert report.n_ingested == 1

    def test_a_renamed_feature_key_fails_loudly(self, tmp_path: Path) -> None:
        """Silently dropping a feature would change the vector without changing a name."""
        broken = {k: v for k, v in FEATURES.items() if k != "heading_entropy"}
        with pytest.raises(ConfigError, match="does not understand"):
            ingest_run(
                run_dir(tmp_path, [decision_record(1, features=broken)]),
                label="bird",
                asserted_by="op",
            )

    def test_a_directory_without_decisions_names_the_mistake(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not at the video"):
            ingest_run(tmp_path, label="bird", asserted_by="op")


class TestStore:
    def test_a_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        report = ingest_run(run_dir(tmp_path, [decision_record(1)]), label="bird", asserted_by="op")
        store = TrackStore(tmp_path / "store.jsonl")
        assert store.append(report.tracks) == 1
        assert store.load() == report.tracks

    def test_re_ingesting_a_run_does_not_double_count_it(self, tmp_path: Path) -> None:
        """Naming the same directory twice is a normal accident, and duplicated rows put
        near-identical samples on both sides of every split."""
        report = ingest_run(
            run_dir(tmp_path, [decision_record(1), decision_record(2)]),
            label="bird",
            asserted_by="op",
        )
        store = TrackStore(tmp_path / "store.jsonl")
        assert store.append(report.tracks) == 2
        assert store.append(report.tracks) == 0
        assert len(store.load()) == 2

    def test_a_corrupt_line_names_its_line_number(self, tmp_path: Path) -> None:
        good = labelled("bird", "b1", 0).to_json()
        path = tmp_path / "store.jsonl"
        path.write_text(f"{good}\nnot json\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"store\.jsonl:2"):
            TrackStore(path).load()

    def test_an_absent_store_is_empty_rather_than_an_error(self, tmp_path: Path) -> None:
        assert TrackStore(tmp_path / "nothing.jsonl").load() == []


class TestCensusCountsClipsNotJustTracks:
    def build(self, tmp_path: Path, spec: list[tuple[str, str, int]]) -> TrackStore:
        store = TrackStore(tmp_path / "store.jsonl")
        for label, clip, count in spec:
            records = [decision_record(i, job=clip) for i in range(count)]
            report = ingest_run(
                run_dir(tmp_path, records, name=clip),
                label=label,
                asserted_by="op",
                source_clip=clip,
            )
            store.append(report.tracks)
        return store

    def test_it_reports_progress_toward_the_target(self, tmp_path: Path) -> None:
        census = self.build(tmp_path, [("bird", "b1", 3), ("drone", "d1", 5)]).census()
        assert census.counts == {"bird": 3, "drone": 5}
        assert census.shortfall(50) == {"bird": 47, "drone": 45}
        assert "needs   47 more" in census.render()

    def test_a_class_from_too_few_clips_is_called_out(self, tmp_path: Path) -> None:
        """Ten tracks from two clips is nearer to two independent samples, and grouped
        folds cannot be formed from them at all."""
        rendered = self.build(tmp_path, [("bird", "b1", 10)]).census().render()
        assert "FEW SOURCE CLIPS" in rendered
        assert "bird" in rendered

    def test_clips_are_counted_separately_from_tracks(self, tmp_path: Path) -> None:
        census = self.build(
            tmp_path, [("bird", "b1", 2), ("bird", "b2", 2), ("bird", "b3", 1)]
        ).census()
        assert census.counts["bird"] == 5
        assert census.clips_per_class["bird"] == 3


def labelled(label: str, clip: str, index: int, **overrides: float) -> LabelledTrack:
    features = {
        "mean_speed": 1.0,
        "speed_variance": 0.1,
        "acceleration_variance": 0.05,
        "vertical_oscillation_hz": 0.2,
        "vertical_oscillation_power": 0.1,
        "heading_entropy": 1.0,
        "hover_fraction": 0.5,
        "trajectory_smoothness": 0.9,
        "scale_change_rate": 0.001,
    }
    features.update(overrides)
    return LabelledTrack(
        track_id=f"{clip}-t{index}",
        source_clip=clip,
        label=label,
        label_basis=LabelBasis.CLIP_PROVENANCE,
        asserted_by="op",
        run_id=clip,
        duration_seconds=5.0,
        n_observations=100,
        median_pixels_on_target=14.0,
        features=features,
    )


class TestFittingRefusesRatherThanOverclaiming:
    def test_ten_tracks_do_not_train_and_the_refusal_is_the_result(self) -> None:
        """The number the operator expects to see, and the correct one. A tree fitted on
        ten samples reports an accuracy that is an artefact of the split."""
        tracks = [labelled("bird", f"b{i}", i) for i in range(3)]
        tracks += [labelled("drone", f"d{i}", i) for i in range(5)]
        tracks += [labelled("aircraft", f"a{i}", i) for i in range(2)]

        outcome = fit_motion_arm(tracks)
        assert outcome.trained is False
        assert outcome.n_tracks == 10
        assert outcome.refusal is not None
        assert "too few" in outcome.refusal
        assert "correct outcome" in outcome.render()

    def test_the_hypothesis_is_undetermined_and_says_which_kind(self) -> None:
        """'Undetermined because no arm ran' is a different claim from 'undetermined
        because the intervals overlapped', and only one of them is about the data."""
        tracks = [labelled("bird", "b1", 0), labelled("drone", "d1", 1)]
        status = hypothesis_status(tracks, fit_motion_arm(tracks))
        assert "UNDETERMINED" in status
        assert "not because the intervals overlapped" in status

    def test_an_empty_store_refuses_without_crashing(self) -> None:
        outcome = fit_motion_arm([])
        assert outcome.trained is False
        assert "empty" in (outcome.refusal or "")


class TestSeparationIsDescriptiveAndSaysSo:
    def test_a_disjoint_feature_is_reported_with_its_chance_rate(self) -> None:
        birds = [
            labelled("bird", f"b{i}", i, vertical_oscillation_hz=4.0 + i * 0.1) for i in range(3)
        ]
        drones = [
            labelled("drone", f"d{i}", i, vertical_oscillation_hz=0.1 + i * 0.05) for i in range(5)
        ]

        report = separate(birds + drones)
        oscillation = next(f for f in report.features if f.feature == "vertical_oscillation_hz")
        assert oscillation.separated_pairs == (("bird", "drone"),)

        rendered = report.render()
        assert "DESCRIPTIVE" in rendered
        assert "expected across" in rendered

    def test_the_chance_rate_is_printed_beside_whatever_was_found(self) -> None:
        """9 features x 1/28 = 0.32 expected separations from noise for a 3-vs-5 split.
        One observed is above that, and the report must say 'preliminary', never
        'result'."""
        birds = [labelled("bird", f"b{i}", i, hover_fraction=0.9 + i * 0.01) for i in range(3)]
        drones = [labelled("drone", f"d{i}", i, hover_fraction=0.1 + i * 0.01) for i in range(5)]
        report = separate(birds + drones)
        assert report.expected_by_chance() == pytest.approx(9 * 2 / 56, rel=1e-6)
        assert len(report.separating) == 1
        rendered = report.render()
        assert "preliminary" in rendered
        assert "never as a result" in rendered

    def test_a_three_class_split_at_these_sizes_expects_three_from_noise(self) -> None:
        """The number that matters for the real dataset: at (3, 5, 2) tracks, about three
        cleanly separated features are expected from noise across nine features. Finding
        three is not a finding."""
        tracks = [labelled("bird", f"b{i}", i) for i in range(3)]
        tracks += [labelled("drone", f"d{i}", i) for i in range(5)]
        tracks += [labelled("aircraft", f"a{i}", i) for i in range(2)]
        assert separate(tracks).expected_by_chance() == pytest.approx(2.98, abs=0.01)

    def test_at_or_below_chance_is_called_out_as_no_evidence(self) -> None:
        tracks = [labelled("bird", f"b{i}", i, mean_speed=1.0 + i) for i in range(3)]
        tracks += [labelled("drone", f"d{i}", i, mean_speed=1.5 + i) for i in range(5)]
        report = separate(tracks)
        assert len(report.separating) == 0
        assert "AT OR BELOW CHANCE" in report.render()

    def test_the_chance_probability_matches_the_combinatorics(self) -> None:
        """2 / C(n_a + n_b, n_a): the two arrangements that keep the groups contiguous."""
        assert chance_separation_probability(3, 5) == pytest.approx(2 / 56)
        assert chance_separation_probability(1, 1) == pytest.approx(1.0)
        assert chance_separation_probability(25, 25) < 1e-13

    def test_an_overlapping_feature_reports_no_separation(self) -> None:
        tracks = [labelled("bird", f"b{i}", i, mean_speed=1.0 + i) for i in range(3)]
        tracks += [labelled("drone", f"d{i}", i, mean_speed=1.5 + i) for i in range(3)]
        report = separate(tracks)
        speed = next(f for f in report.features if f.feature == "mean_speed")
        assert speed.separated_pairs == ()


class TestTheStoreStaysOutsideTheRepository:
    """Derived from footage whose licence this project does not control (CLAUDE.md 4)."""

    def test_the_default_store_path_is_outside_the_repo(self) -> None:
        from tayr.cli.main import DEFAULT_TRACK_STORE

        assert str(DEFAULT_TRACK_STORE).startswith(".."), DEFAULT_TRACK_STORE

    def test_an_in_repo_store_would_still_be_gitignored(self) -> None:
        """Intent is not a control. Someone will pass --store tracks.jsonl one day."""
        ignored = Path(".gitignore").read_text(encoding="utf-8")
        assert "*.jsonl" in ignored
        assert "demo-runs/" in ignored
