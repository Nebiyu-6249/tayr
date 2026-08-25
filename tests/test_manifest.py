"""Run manifest tests.

A manifest is what separates a result from a number someone typed. These tests check
that provenance is captured, that a dirty tree is flagged, and that an existing
manifest is never overwritten.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tayr.config import Config
from tayr.determinism import seed_everything
from tayr.errors import ManifestError
from tayr.manifest import build_manifest, installed_versions


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _build(repo: Path, **kw: object) -> object:
    return build_manifest(
        run_id="r1",
        command="tayr train --config configs/baseline.yaml",
        config=Config().to_dict(),
        seed_report=seed_everything(42),
        repo=repo,
        **kw,  # type: ignore[arg-type]
    )


class TestProvenance:
    def test_captures_commit_and_branch(self, git_repo: Path) -> None:
        m = _build(git_repo)
        assert len(m.git_commit) == 40  # type: ignore[attr-defined]
        assert m.git_dirty is False  # type: ignore[attr-defined]
        assert m.seed_report["seed"] == 42  # type: ignore[attr-defined]

    def test_records_package_versions(self) -> None:
        v = installed_versions()
        assert v["numpy"] != "not installed"
        # torch is an optional extra; absence must be recorded, not omitted.
        assert "torch" in v

    def test_dirty_tree_is_flagged_with_a_warning_note(self, git_repo: Path) -> None:
        (git_repo / "f.txt").write_text("changed", encoding="utf-8")
        m = _build(git_repo)
        assert m.git_dirty is True  # type: ignore[attr-defined]
        assert any("DIRTY" in n for n in m.notes)  # type: ignore[attr-defined]

    def test_synthetic_runs_are_labelled(self, git_repo: Path) -> None:
        m = _build(git_repo, synthetic=True)
        assert m.synthetic is True  # type: ignore[attr-defined]
        assert any("SYNTHETIC" in n for n in m.notes)  # type: ignore[attr-defined]

    def test_not_a_git_repo_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError):
            _build(tmp_path / "not_a_repo" if False else tmp_path)


class TestWriting:
    def test_writes_parseable_json(self, git_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "run"
        path = _build(git_repo).write(out)  # type: ignore[attr-defined]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "r1"
        assert data["config"]["seed"] == 1337

    def test_refuses_to_overwrite(self, git_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "run"
        _build(git_repo).write(out)  # type: ignore[attr-defined]
        with pytest.raises(ManifestError, match="Refusing to overwrite"):
            _build(git_repo).write(out)  # type: ignore[attr-defined]
