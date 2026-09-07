"""Training wiring.

The test that matters most here is `TestUpstreamContract`. Everything else checks our
own logic; that one checks that the names we send to RF-DETR are still the names RF-DETR
takes. A hyperparameter that quietly reverts to a default mid-run is the kind of bug that
produces a result nobody can explain three weeks later, and it is invisible without this.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tayr.config import Config
from tayr.errors import ConfigError, ManifestError
from tayr.train.detector import (
    ABSORBED_KWARGS,
    DATASET_FILE,
    LIGHTWEIGHT_CHECKPOINTS,
    new_run_id,
    rfdetr_train_kwargs,
    train_detector,
)
from tests.cv_extra import requires_cv_extra

REPO = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **overrides: object) -> Config:
    payload: dict[str, object] = {
        "seed": 5,
        "device": "cpu",
        "output_dir": str(tmp_path / "runs"),
        "train": {"dataset_dir": str(tmp_path / "ds"), "epochs": 3, "batch_size": 2},
    }
    payload.update(overrides)
    return Config.model_validate(payload)


class TestKwargMapping:
    def test_every_tayr_key_reaches_rfdetr_under_its_own_name(self, tmp_path: Path) -> None:
        cfg = make_config(
            tmp_path,
            train={
                "dataset_dir": str(tmp_path / "ds"),
                "epochs": 12,
                "batch_size": 3,
                "grad_accum_steps": 7,
                "learning_rate": 0.002,
                "lr_encoder": 0.003,
                "weight_decay": 0.05,
                "warmup_epochs": 1.5,
                "num_workers": 6,
                "checkpoint_interval": 4,
                "eval_interval": 2,
                "early_stopping": True,
                "early_stopping_patience": 9,
                "use_ema": False,
                "tensorboard": True,
                "resolution": 448,
            },
        )
        kwargs = rfdetr_train_kwargs(cfg, device="cuda:0", output_dir=tmp_path / "out")
        assert kwargs["epochs"] == 12
        assert kwargs["batch_size"] == 3
        assert kwargs["grad_accum_steps"] == 7
        # Tayr calls it learning_rate; RF-DETR calls it lr. This is the rename that
        # would otherwise be silent.
        assert kwargs["lr"] == 0.002
        assert kwargs["lr_encoder"] == 0.003
        assert kwargs["weight_decay"] == 0.05
        assert kwargs["warmup_epochs"] == 1.5
        assert kwargs["num_workers"] == 6
        assert kwargs["checkpoint_interval"] == 4
        assert kwargs["eval_interval"] == 2
        assert kwargs["early_stopping"] is True
        assert kwargs["early_stopping_patience"] == 9
        assert kwargs["use_ema"] is False
        assert kwargs["tensorboard"] is True
        assert kwargs["resolution"] == 448
        assert kwargs["device"] == "cuda:0"
        assert kwargs["dataset_file"] == DATASET_FILE

    def test_external_loggers_are_disabled_explicitly(self, tmp_path: Path) -> None:
        """Not left to an upstream default that a future release could flip."""
        kwargs = rfdetr_train_kwargs(make_config(tmp_path), device="cpu", output_dir=tmp_path)
        assert kwargs["wandb"] is False
        assert kwargs["mlflow"] is False
        assert kwargs["clearml"] is False

    def test_optional_keys_are_omitted_rather_than_sent_as_none(self, tmp_path: Path) -> None:
        """RF-DETR's TrainConfig forbids extras; `resolution=None` would override a
        variant's own default with nothing."""
        kwargs = rfdetr_train_kwargs(make_config(tmp_path), device="cpu", output_dir=tmp_path)
        assert "resolution" not in kwargs
        assert "resume" not in kwargs

    def test_missing_dataset_dir_names_the_key(self, tmp_path: Path) -> None:
        cfg = Config.model_validate({"output_dir": str(tmp_path)})
        with pytest.raises(ConfigError, match=re.escape("train.dataset_dir")):
            rfdetr_train_kwargs(cfg, device="cpu", output_dir=tmp_path)

    def test_mapping_needs_no_torch(self, tmp_path: Path) -> None:
        """Pure by construction, so a config can be checked on a machine with no cv extra."""
        kwargs = rfdetr_train_kwargs(make_config(tmp_path), device="cpu", output_dir=tmp_path)
        assert kwargs["output_dir"] == str(tmp_path)


class TestUpstreamContract:
    """Every kwarg we send must still exist upstream."""

    def test_kwargs_are_all_real_rfdetr_fields(self, tmp_path: Path) -> None:
        rfdetr_config = pytest.importorskip(
            "rfdetr.config", reason="rfdetr is in the cv extra; skip where it is not installed"
        )
        fields = set(rfdetr_config.TrainConfig.model_fields)
        kwargs = rfdetr_train_kwargs(
            make_config(
                tmp_path,
                train={
                    "dataset_dir": str(tmp_path / "ds"),
                    "resolution": 448,
                    "resume_from": str(tmp_path / "last.ckpt"),
                },
            ),
            device="cpu",
            output_dir=tmp_path,
        )
        unknown = set(kwargs) - fields - ABSORBED_KWARGS
        assert not unknown, (
            f"these kwargs are neither TrainConfig fields nor absorbed by train(): "
            f"{sorted(unknown)}. RF-DETR sets extra='forbid', so passing one would raise "
            "at run time, hours into a training session."
        )

    def test_absorbed_kwargs_are_genuinely_not_fields(self) -> None:
        """If an absorbed kwarg becomes a real field upstream, the exemption is stale."""
        rfdetr_config = pytest.importorskip("rfdetr.config")
        overlap = ABSORBED_KWARGS & set(rfdetr_config.TrainConfig.model_fields)
        assert not overlap, (
            f"{sorted(overlap)} are now TrainConfig fields upstream; remove them from "
            "ABSORBED_KWARGS so the contract test actually checks them."
        )

    def test_lightweight_checkpoint_names_still_exist_upstream(self) -> None:
        """The resume warning is keyed on these filenames; a rename would mute it."""
        detr = pytest.importorskip("rfdetr.detr")
        source = Path(detr.__file__).read_text(encoding="utf-8")
        for name in LIGHTWEIGHT_CHECKPOINTS:
            assert name in source, f"{name} no longer appears in rfdetr/detr.py"


class TestRunIdentity:
    def test_run_ids_sort_chronologically(self) -> None:
        early = new_run_id("nano", now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
        late = new_run_id("nano", now=datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC))
        assert early < late
        assert early == "20260102-030405-rfdetr-nano"


@requires_cv_extra
class TestDryRun:
    """Seeding is strict at training entry points, so even a dry run needs torch."""

    def test_manifest_is_written_before_any_training(self, tmp_path: Path) -> None:
        run = train_detector(
            make_config(tmp_path), run_id="r", repo=REPO, synthetic=True, dry_run=True
        )
        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "r"
        assert manifest["synthetic"] is True
        assert any("SYNTHETIC DATA" in note for note in manifest["notes"])
        assert any("DRY RUN" in note for note in manifest["notes"])

    def test_device_resolution_is_recorded(self, tmp_path: Path) -> None:
        run = train_detector(make_config(tmp_path), run_id="r", repo=REPO, dry_run=True)
        assert run.device.device == "cpu"
        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert any("device:" in note for note in manifest["notes"])

    def test_a_second_run_into_the_same_directory_refuses(self, tmp_path: Path) -> None:
        """Overwriting a manifest destroys the provenance of the result beside it."""
        cfg = make_config(tmp_path)
        train_detector(cfg, run_id="r", repo=REPO, dry_run=True)
        with pytest.raises(ManifestError, match="Refusing to overwrite"):
            train_detector(cfg, run_id="r", repo=REPO, dry_run=True)

    def test_resuming_writes_a_separate_manifest_instead(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        first = train_detector(cfg, run_id="r", repo=REPO, dry_run=True)
        resumed = train_detector(
            make_config(
                tmp_path,
                train={
                    "dataset_dir": str(tmp_path / "ds"),
                    "resume_from": str(tmp_path / "runs" / "r" / "last.ckpt"),
                },
            ),
            run_id="r",
            repo=REPO,
            dry_run=True,
        )
        assert first.manifest_path.name == "manifest.json"
        assert resumed.manifest_path.name == "manifest.resume-001.json"
        assert first.manifest_path.exists()

    def test_resuming_from_a_lightweight_checkpoint_warns_in_the_manifest(
        self, tmp_path: Path
    ) -> None:
        """Those omit optimizer state, so the loss curve is not continuous across the resume."""
        run = train_detector(
            make_config(
                tmp_path,
                train={
                    "dataset_dir": str(tmp_path / "ds"),
                    "resume_from": str(tmp_path / "checkpoint_best_total.pth"),
                },
            ),
            run_id="r",
            repo=REPO,
            dry_run=True,
        )
        notes = json.loads(run.manifest_path.read_text(encoding="utf-8"))["notes"]
        assert any("restart cold" in note for note in notes)

    def test_an_unimplemented_backend_is_refused_not_silently_swapped(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, detector={"backend": "dfine"})
        with pytest.raises(ConfigError, match="no training implementation"):
            train_detector(cfg, run_id="r", repo=REPO, dry_run=True)
