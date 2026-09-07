"""CLI tests. The CLI is a thin wrapper, so these check wiring and exit codes only."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tayr import __version__
from tayr.cli.main import app
from tests.cv_extra import requires_cv_extra

runner = CliRunner()
BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_validate_prints_resolved_config() -> None:
    result = runner.invoke(app, ["config", "validate", "--config", str(BASELINE)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["detector"]["backend"] == "rfdetr"


def test_config_validate_exits_nonzero_on_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("slicing:\n  overlap: 2.0\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", "--config", str(bad)])
    assert result.exit_code == 1


@requires_cv_extra
def test_train_refuses_a_config_with_no_dataset() -> None:
    """The baseline config names no detector dataset, so training has nothing to read.

    This is the shape of every "you forgot to set X" failure: a clean non-zero exit
    naming the key, not a traceback and not a run directory full of nothing.
    """
    result = runner.invoke(app, ["train", "--config", str(BASELINE), "--dry-run"])
    assert result.exit_code == 1
    assert "train.dataset_dir" in result.output


@requires_cv_extra
def test_train_dry_run_writes_a_manifest_without_training(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(
        "seed: 7\n"
        "device: cpu\n"
        f"output_dir: {tmp_path / 'runs'}\n"
        "train:\n"
        f"  dataset_dir: {tmp_path / 'ds'}\n"
        "  epochs: 1\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(config), "--dry-run", "--run-id", "r1"])
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output

    manifest = json.loads((tmp_path / "runs" / "r1" / "manifest.json").read_text())
    assert manifest["seed_report"]["seed"] == 7
    assert any("DRY RUN" in note for note in manifest["notes"])
    assert not list((tmp_path / "runs" / "r1").glob("*.pth"))


def test_eval_refuses_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    """`tayr eval` reads its config from the run's manifest, so a bare directory fails."""
    result = runner.invoke(app, ["eval", "--run", str(tmp_path)])
    assert result.exit_code == 1
    assert "manifest.json" in result.output
