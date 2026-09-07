"""Run manifests.

Every training or evaluation run writes one of these next to its outputs. It records
what was run, from which commit, with which seed, against which library versions. A
result without a manifest is not a result - it cannot be reproduced or audited.

Two fields exist specifically to stop the project lying to itself:

  `git_dirty`   - True if the working tree had uncommitted changes. A run from a dirty
                  tree is not reproducible and any number it produces must be labelled
                  provisional.
  `synthetic`   - True if the run consumed placeholder or synthetic data. Set this and
                  the label propagates into logs, reports and the UI.
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from tayr.determinism import SeedReport
from tayr.errors import ManifestError

# Recorded in every manifest. Versions of things that can change a numeric result.
_TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "rfdetr",
    "sahi",
    "opencv-python",
    "scikit-learn",
    "xgboost",
    "av",
)


def _git(*args: str, repo: Path) -> str:
    """Run a git command, raising ManifestError with context if it fails."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise ManifestError("git executable not found; cannot build a run manifest") from exc
    except subprocess.CalledProcessError as exc:
        raise ManifestError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ManifestError(f"git {' '.join(args)} timed out") from exc
    return out.stdout.strip()


def installed_versions() -> dict[str, str]:
    """Version of each tracked package, or 'not installed'."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything needed to understand where a number came from."""

    run_id: str
    command: str
    created_at: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    config: dict[str, Any]
    config_path: str | None
    seed_report: dict[str, Any]
    python_version: str
    platform: str
    hostname: str
    user: str
    package_versions: dict[str, str]
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, directory: Path, *, filename: str = "manifest.json") -> Path:
        """Write a manifest into `directory`, refusing to overwrite an existing one.

        `filename` exists for resumed runs. A resume continues into the same run
        directory - RF-DETR's best-score tracking only restores when `output_dir` matches
        the checkpoint's original directory - but it is a separate launch, on a possibly
        different commit and a different machine, so it gets its own manifest rather than
        overwriting or silently sharing the first one's provenance.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        if path.exists():
            raise ManifestError(
                f"{path} already exists. Refusing to overwrite a run manifest - "
                "that would destroy the provenance of an existing result."
            )
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def build_manifest(
    *,
    run_id: str,
    command: str,
    config: dict[str, Any],
    seed_report: SeedReport,
    config_path: Path | None = None,
    repo: Path | None = None,
    synthetic: bool = False,
    notes: list[str] | None = None,
) -> RunManifest:
    """Collect provenance for a run. Raises if git metadata is unavailable."""
    repo = repo or Path.cwd()
    commit = _git("rev-parse", "HEAD", repo=repo)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo)
    dirty = bool(_git("status", "--porcelain", repo=repo))

    all_notes = list(notes or [])
    if dirty:
        all_notes.append(
            "WORKING TREE DIRTY at run time. This run is not reproducible from the "
            "recorded commit alone; treat any resulting number as provisional."
        )
    if synthetic:
        all_notes.append(
            "SYNTHETIC DATA. This run consumed placeholder or synthetic data. No number "
            "in it describes real-world performance."
        )

    return RunManifest(
        run_id=run_id,
        command=command,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        config=config,
        config_path=str(config_path) if config_path else None,
        seed_report=asdict(seed_report),
        python_version=sys.version,
        platform=platform.platform(),
        hostname=socket.gethostname(),
        user=getpass.getuser(),
        package_versions=installed_versions(),
        synthetic=synthetic,
        notes=all_notes,
    )
