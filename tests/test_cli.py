"""CLI tests. The CLI is a thin wrapper, so these check wiring and exit codes only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tayr import __version__
from tayr.cli.main import app

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


@pytest.mark.parametrize("cmd", [["train", "--config"], ["eval", "--run"]])
def test_unimplemented_commands_raise_rather_than_return_placeholders(
    cmd: list[str], tmp_path: Path
) -> None:
    """Phase 3 implements these. Until then they must fail loudly, not return a stub."""
    arg = str(BASELINE) if cmd[0] == "train" else str(tmp_path)
    result = runner.invoke(app, [*cmd, arg])
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "Phase 3" in str(result.exception)
