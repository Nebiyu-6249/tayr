"""Tayr CLI.

Owner-mode entry point. Every command is a thin wrapper: it parses arguments, calls
into the `tayr` library, and prints. No business logic lives in this module.

Commands that are not yet implemented raise loudly and name the phase that will
implement them. They do not return placeholder results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tayr import __version__
from tayr.config import load_config
from tayr.datasets.census import take_census
from tayr.datasets.coco import to_coco
from tayr.datasets.loader import load_native_directory
from tayr.errors import TayrError

app = typer.Typer(
    name="tayr",
    help="Detection, tracking and motion classification of small aerial objects in recorded video.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    name="config", help="Inspect and validate run configs.", no_args_is_help=True
)
app.add_typer(config_app)

dataset_app = typer.Typer(
    name="dataset", help="Convert and measure datasets.", no_args_is_help=True
)
app.add_typer(dataset_app)

ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Path to a run config YAML.")]


@app.command()
def version() -> None:
    """Print the Tayr version."""
    typer.echo(__version__)


@config_app.command("validate")
def config_validate(config: ConfigPath) -> None:
    """Validate a run config against the schema and print the resolved values.

    Unknown keys are rejected, so this also catches typos that would otherwise fall
    back to a default and silently change a result.
    """
    try:
        cfg = load_config(config)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))


@dataset_app.command("census")
def dataset_census(
    dataset: Annotated[Path, typer.Option("--dataset", help="Directory of annotation files.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: dvb | antiuav.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name for the report.")] = "unnamed",
    split: Annotated[str, typer.Option("--split", help="Split name for the report.")] = "all",
) -> None:
    """Count what a dataset actually contains: videos, frames, boxes, and TRACKS.

    Run this before writing any training code. The motion classifier trains on tracks,
    and a dataset advertising thousands of images may contain none.
    """
    try:
        annotated = load_native_directory(dataset, fmt=fmt, name=name, split=split)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(take_census(annotated).render())


@dataset_app.command("convert")
def dataset_convert(
    dataset: Annotated[Path, typer.Option("--dataset", help="Directory of annotation files.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: dvb | antiuav.")],
    out: Annotated[Path, typer.Option("--out", help="Output COCO JSON path.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name.")] = "unnamed",
    split: Annotated[str, typer.Option("--split", help="Split name.")] = "all",
) -> None:
    """Convert native annotations to COCO JSON.

    The output may contain annotations derived from a dataset that grants no
    redistribution rights. Write it outside the repository and never commit it.
    """
    try:
        annotated = load_native_directory(dataset, fmt=fmt, name=name, split=split)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_coco(annotated), indent=2), encoding="utf-8")

    census = take_census(annotated)
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"  {census.n_videos} video(s), {census.n_frames} frame(s), "
        f"{census.n_boxes} box(es), {census.n_tracks} track(s)"
    )
    if not annotated.redistributable:
        typer.secho(
            "  NOTE: this dataset is marked non-redistributable. Do not commit this "
            "file or place it anywhere public.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def train(config: ConfigPath) -> None:
    """Train a detector from a config. [Phase 3 - not yet implemented]"""
    load_config(config)  # validate now so a bad config fails before anything else
    raise NotImplementedError(
        "tayr train is implemented in Phase 3 (detection training pipeline). The config "
        "you passed is valid, but no training code exists yet. This command will not "
        "return a placeholder result."
    )


@app.command("eval")
def evaluate(
    run: Annotated[Path, typer.Option("--run", help="Run directory produced by `tayr train`.")],
    split: Annotated[str, typer.Option("--split", help="Dataset split to evaluate.")] = "test",
) -> None:
    """Evaluate a trained run. [Phase 3 - not yet implemented]"""
    raise NotImplementedError(
        f"tayr eval is implemented in Phase 3 (evaluation harness). Requested run={run} "
        f"split={split}. No evaluation code exists yet, and this command will not return "
        "a fabricated metric."
    )


if __name__ == "__main__":
    app()
