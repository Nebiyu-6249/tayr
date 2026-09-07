"""Loader and dataset-CLI tests, exercised against on-disk fixtures in tmp_path.

No dataset file is committed; every fixture is written by the test itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tayr.cli.main import app
from tayr.datasets.loader import load_native_directory
from tayr.errors import ConfigError

runner = CliRunner()


@pytest.fixture
def dvb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dvb"
    d.mkdir()
    (d / "clip_a.txt").write_text("0 1 100 200 30 40 drone\n1 0\n", encoding="utf-8")
    (d / "clip_b.txt").write_text("0 2 10 10 5 5 drone 60 60 4 4 bird\n", encoding="utf-8")
    return d


@pytest.fixture
def antiuav_dir(tmp_path: Path) -> Path:
    d = tmp_path / "antiuav"
    for name in ("seq01", "seq02"):
        sub = d / name
        sub.mkdir(parents=True)
        (sub / "IR_label.json").write_text(
            json.dumps({"gt_rect": [[10, 20, 30, 40], []], "exist": [1, 0]}), encoding="utf-8"
        )
    return d


class TestLoader:
    def test_loads_dvb_directory(self, dvb_dir: Path) -> None:
        ds = load_native_directory(dvb_dir, fmt="dvb", name="dvb", split="train")
        assert ds.n_videos == 2
        assert {v.source_video for v in ds.videos} == {"clip_a", "clip_b"}
        assert ds.n_boxes == 3

    def test_loads_antiuav_directory(self, antiuav_dir: Path) -> None:
        ds = load_native_directory(antiuav_dir, fmt="antiuav", name="antiuav", split="test")
        assert ds.n_videos == 2
        assert ds.n_frames == 4
        assert ds.n_boxes == 2  # one present frame per sequence

    def test_each_antiuav_sequence_gets_a_distinct_track_id(self, antiuav_dir: Path) -> None:
        ds = load_native_directory(antiuav_dir, fmt="antiuav", name="a", split="test")
        ids = {o.track_id for v in ds.videos for f in v.frames for o in f.objects}
        assert ids == {0, 1}

    def test_dvb_is_marked_non_redistributable(self, dvb_dir: Path) -> None:
        """The data usage agreement grants no redistribution rights, and that fact must
        travel with the data rather than living only in a person's memory."""
        ds = load_native_directory(dvb_dir, fmt="dvb", name="dvb", split="train")
        assert ds.redistributable is False
        assert "no redistribution" in ds.licence

    def test_the_dut_format_name_points_at_the_right_subset(self, tmp_path: Path) -> None:
        """`dut` is the name a person tries first, and it is genuinely ambiguous.

        The detection subset is VOC XML and is supported; the tracking subset ships one
        first-frame box per video and has no tracks to convert. The error has to say
        which is which rather than just listing valid names.
        """
        with pytest.raises(ConfigError, match="there is no 'dut' format"):
            load_native_directory(tmp_path, fmt="dut", name="x", split="y")

    def test_an_unknown_format_lists_the_supported_ones(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unsupported format"):
            load_native_directory(tmp_path, fmt="parquet", name="x", split="y")

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a directory"):
            load_native_directory(tmp_path / "nope", fmt="dvb", name="x", split="y")

    def test_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=re.escape("no *.txt annotation files")):
            load_native_directory(tmp_path, fmt="dvb", name="x", split="y")


class TestDatasetCli:
    def test_census_command(self, dvb_dir: Path) -> None:
        result = runner.invoke(
            app, ["dataset", "census", "--dataset", str(dvb_dir), "--format", "dvb"]
        )
        assert result.exit_code == 0, result.output
        assert "TRACKS per class" in result.stdout
        assert "ZERO tracks" in result.stdout

    def test_convert_command_writes_coco(self, dvb_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out" / "coco.json"
        result = runner.invoke(
            app,
            ["dataset", "convert", "--dataset", str(dvb_dir), "--format", "dvb", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        coco = json.loads(out.read_text(encoding="utf-8"))
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) == 3

    def test_convert_warns_about_redistribution(self, dvb_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "dataset",
                "convert",
                "--dataset",
                str(dvb_dir),
                "--format",
                "dvb",
                "--out",
                str(tmp_path / "c.json"),
            ],
        )
        assert "non-redistributable" in result.stdout

    def test_census_exits_nonzero_on_bad_input(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["dataset", "census", "--dataset", str(tmp_path / "nope"), "--format", "dvb"]
        )
        assert result.exit_code == 1
