"""Config tests.

The point of the schema is that a bad config fails at load time with a clear message,
rather than falling back to a default and quietly changing a result weeks later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tayr.config import Config, load_config
from tayr.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


class TestLoading:
    def test_repo_baseline_config_is_valid(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml")
        assert cfg.seed == 1337
        assert cfg.detector.backend == "rfdetr"
        # Named for the subset, not the dataset: DUT ships a detection subset (VOC XML,
        # supported) and a tracking subset (one first-frame box per video, nothing to
        # convert). Conflating them is the mistake this name exists to prevent.
        assert cfg.datasets[0].name == "dut-anti-uav-detection"
        assert cfg.datasets[0].annotation_format == "voc"

    def test_defaults_apply_to_minimal_config(self, tmp_path: Path) -> None:
        cfg = load_config(_write(tmp_path, "seed: 7\n"))
        assert cfg.seed == 7
        assert cfg.slicing.tile_width == 640

    def test_round_trips_through_dict(self) -> None:
        cfg = Config()
        assert Config.model_validate(cfg.to_dict()) == cfg


class TestRejects:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="is empty"):
            load_config(_write(tmp_path, ""))

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(_write(tmp_path, "seed: [1, 2\n"))

    def test_top_level_not_a_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_config(_write(tmp_path, "- a\n- b\n"))

    def test_unknown_key_is_an_error_not_a_silent_default(self, tmp_path: Path) -> None:
        """A typo must fail. This is the whole reason extra='forbid' is set."""
        with pytest.raises(ConfigError, match="learing_rate"):
            load_config(_write(tmp_path, "train:\n  learing_rate: 0.1\n"))

    def test_out_of_range_value(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="overlap"):
            load_config(_write(tmp_path, "slicing:\n  overlap: 1.5\n"))


class TestInvariants:
    """Cross-field rules that encode decisions from the research phase."""

    def test_tracker_thresholds_must_be_ordered(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="low_threshold"):
            load_config(_write(tmp_path, "tracker:\n  high_threshold: 0.2\n  low_threshold: 0.5\n"))

    def test_ungrouped_splits_refused(self, tmp_path: Path) -> None:
        """Ungrouped splits leak frames across train/test and inflate every number."""
        with pytest.raises(ConfigError, match="group_splits_by_video"):
            load_config(_write(tmp_path, "eval:\n  group_splits_by_video: false\n"))

    def test_checkpoint_requires_checksum(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="checkpoint_sha256"):
            load_config(_write(tmp_path, "detector:\n  checkpoint: weights/x.pt\n"))

    def test_checkpoint_with_checksum_is_accepted(self, tmp_path: Path) -> None:
        cfg = load_config(
            _write(
                tmp_path,
                "detector:\n  checkpoint: weights/x.pt\n  checkpoint_sha256: abc123\n",
            )
        )
        assert cfg.detector.checkpoint_sha256 == "abc123"

    def test_dataset_redistributable_defaults_to_false(self, tmp_path: Path) -> None:
        """Defaulting to False means forgetting the flag cannot cause a licence breach."""
        cfg = load_config(
            _write(
                tmp_path,
                "datasets:\n  - name: x\n    root: data/x\n    annotation_format: coco\n",
            )
        )
        assert cfg.datasets[0].redistributable is False
        assert cfg.datasets[0].licence == "UNKNOWN"
